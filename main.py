import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="UED Detection Rover ML Pipeline Unified Entry Point")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # 1. generate
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
    gen_parser.add_argument('--output_dir', type=str, default='./data')
    gen_parser.add_argument('--seed', type=int, default=42)

    # 2. train
    train_parser = subparsers.add_parser('train', help='Train HMLFNet model')
    train_parser.add_argument('--data_dir', type=str, default='./data')
    train_parser.add_argument('--output_dir', type=str, default='./results')
    train_parser.add_argument('--epochs', type=int, default=60)
    train_parser.add_argument('--fast_debug', action='store_true')

    # 3. evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--data_dir', type=str, default='./data')
    eval_parser.add_argument('--model_path', type=str, default='./results/best_model.pt')

    # 4. export
    export_parser = subparsers.add_parser('export', help='Export model to ONNX and INT8 Quantised PT')
    export_parser.add_argument('--model_path', type=str, default='./results/best_model.pt')
    export_parser.add_argument('--output_dir', type=str, default='./model_export')

    # 5. visualise
    vis_parser = subparsers.add_parser('visualise', help='Run interactive visualisation dashboard')
    vis_parser.add_argument('--data_dir', type=str, default='./data')
    vis_parser.add_argument('--model_path', type=str, default='./model_export/hmlf_net.pt')
    vis_parser.add_argument('--sample_idx', type=int, default=42)
    vis_parser.add_argument('--random', action='store_true')

    # 6. animate
    anim_parser = subparsers.add_parser('animate', help='Run batch scanning animation')
    anim_parser.add_argument('--data_dir', type=str, default='./data')
    anim_parser.add_argument('--model_path', type=str, default='./model_export/hmlf_net.pt')

    # 7. pipeline
    pipe_parser = subparsers.add_parser('pipeline', help='Run the entire end-to-end pipeline')
    pipe_parser.add_argument('--seed', type=int, default=42)
    pipe_parser.add_argument('--fast_debug', action='store_true')

    args = parser.parse_args()

    if args.command == 'generate':
        import generate_dataset
        generate_dataset.main(args)

    elif args.command == 'train':
        import train_model
        train_model.main(args)

    elif args.command == 'evaluate':
        import torch
        from dataset import UEDDataset
        from torch.utils.data import DataLoader
        from model.hmlf_net import HMLFNet
        from evaluate import evaluate_model
        import os

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_ds = UEDDataset(os.path.join(args.data_dir, 'test.npz'), os.path.join(args.data_dir, 'dataset_stats.json'), split='test')
        test_loader = DataLoader(test_ds, batch_size=64)

        model = HMLFNet()
        # Handle fallback if model_path isn't exact (as train_model saves to model_export currently, but prompt specifies ./results/best_model.pt)
        if not os.path.exists(args.model_path) and os.path.exists('./model_export/hmlf_net.pt'):
            args.model_path = './model_export/hmlf_net.pt'

        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        evaluate_model(model, test_loader, device, './results')

    elif args.command == 'export':
        import torch
        from model.hmlf_net import HMLFNet
        from export import export_model
        import os

        model = HMLFNet()
        if not os.path.exists(args.model_path) and os.path.exists('./model_export/hmlf_net.pt'):
            args.model_path = './model_export/hmlf_net.pt'

        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        export_model(model, args.output_dir)

    elif args.command == 'visualise':
        import visualise
        visualise.main(args)

    elif args.command == 'animate':
        import visualise
        args.animate_only = True
        visualise.main(args)

    elif args.command == 'pipeline':
        print("Running full pipeline end-to-end...")

        print("\n--- STEP 1: GENERATE DATASET ---")
        import generate_dataset
        # Create args compatible with generate
        gen_args = argparse.Namespace(output_dir='./data', seed=args.seed)
        generate_dataset.main(gen_args)

        print("\n--- STEP 2: TRAIN MODEL ---")
        import train_model
        train_args = argparse.Namespace(data_dir='./data', output_dir='./results', epochs=60 if not args.fast_debug else 2, fast_debug=args.fast_debug)
        train_model.main(train_args)

        print("\n--- STEP 3: VISUALISE (Panel 1-4) ---")
        import visualise
        vis_args = argparse.Namespace(sample_idx=42, random=False, animate_only=False)
        visualise.main(vis_args)

if __name__ == '__main__':
    main()
