import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
import math
import copy

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, epochs=60):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)

        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_loc = nn.SmoothL1Loss(reduction='none')
        self.criterion_conf = nn.BCELoss()

    def get_lambda_loc(self, epoch):
        if epoch < 5:
            return 0.1 + (0.4 * (epoch / 5.0))
        return 0.5

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        lambda_loc = self.get_lambda_loc(epoch)

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]"):
            mag = batch['mag'].to(self.device)
            thermal = batch['thermal'].to(self.device)
            gas = batch['gas'].to(self.device)
            labels = batch['label'].to(self.device)
            meta = batch['metadata'].to(self.device) # shape: (B, 4)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, loc, conf, _ = self.model(mag, thermal, gas)

                    # Classification loss
                    l_cls = self.criterion_cls(logits, labels)

                    # Localisation loss
                    # meta has [row_mag, col_mag, row_thermal, col_thermal]
                    # We need to normalize target to [0, 1] as per prompt
                    target_meta = meta.clone()
                    target_meta[:, 0] /= 4.0 # mag_row max is 4
                    target_meta[:, 1] /= 4.0 # mag_col max is 4
                    target_meta[:, 2] /= 7.0 # thermal_row max is 7
                    target_meta[:, 3] /= 7.0 # thermal_col max is 7

                    target_meta_safe = target_meta.clone()
                    target_meta_safe[torch.isnan(target_meta_safe)] = 0.0

                    l_loc_all = self.criterion_loc(loc, target_meta_safe)
                    mask = (labels != 0).float().unsqueeze(1) # Mask out class 0
                    l_loc = (l_loc_all * mask).sum() / (mask.sum() * 4 + 1e-8)

                    # Confidence loss
                    preds = torch.argmax(logits, dim=1)
                    conf_target = (preds == labels).float().unsqueeze(1).detach()
                    conf_clamped = torch.clamp(conf, 1e-6, 1.0 - 1e-6).float()
                    l_conf = self.criterion_conf(conf_clamped, conf_target)

                    loss = l_cls + lambda_loc * l_loc + 0.1 * l_conf

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, loc, conf, _ = self.model(mag, thermal, gas)
                l_cls = self.criterion_cls(logits, labels)

                target_meta = meta.clone()
                target_meta[:, 0] /= 4.0
                target_meta[:, 1] /= 4.0
                target_meta[:, 2] /= 7.0
                target_meta[:, 3] /= 7.0

                # Mask NaNs in target_meta so SmoothL1Loss doesn't propagate NaNs
                target_meta_safe = target_meta.clone()
                target_meta_safe[torch.isnan(target_meta_safe)] = 0.0

                l_loc_all = self.criterion_loc(loc, target_meta_safe)
                mask = (labels != 0).float().unsqueeze(1)
                l_loc = (l_loc_all * mask).sum() / (mask.sum() * 4 + 1e-8)

                preds = torch.argmax(logits, dim=1)
                conf_target = (preds == labels).float().unsqueeze(1).detach()

                # Squeeze the confidence output to ensure it matches target size and is in bounds
                # Actually, conf output may be slightly out of bounds due to numerical issues if not clamped
                # Let's use BCEWithLogitsLoss or just clamp if it's already sigmoid
                conf_clamped = torch.clamp(conf, 1e-6, 1.0 - 1e-6).float()
                l_conf = self.criterion_conf(conf_clamped, conf_target)

                loss = l_cls + lambda_loc * l_loc + 0.1 * l_conf

                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="[Validate]"):
                mag = batch['mag'].to(self.device)
                thermal = batch['thermal'].to(self.device)
                gas = batch['gas'].to(self.device)
                labels = batch['label'].to(self.device)
                meta = batch['metadata'].to(self.device)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits, loc, conf, _ = self.model(mag, thermal, gas)
                        l_cls = self.criterion_cls(logits, labels)
                else:
                    logits, loc, conf, _ = self.model(mag, thermal, gas)
                    l_cls = self.criterion_cls(logits, labels)

                total_loss += l_cls.item()
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(self.val_loader), f1_macro

    def fit(self):
        best_f1 = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_f1 = self.validate()

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1-Macro: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= 10:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.model
