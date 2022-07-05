import argparse


def get_arguments():
    """
    Parsing arguments
    """
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='monumai',
                        choices=['pascal', 'monumai'], help='Name of dataset')
    # Deeplab Options
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=5000,  # 30e3
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr_deeplab", type=float, default=0.01,
                        help="learning rate of Deeplab(default: 0.01)")
    parser.add_argument("--lr_logreg", type=float, default=0.01,
                        help="learning rate of lr_logreg (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--crop_size", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--class_weights", type=str, default="enet",
                        choices=['enet', 'none'], help="Add classes weight in the Cross Entropy")
    parser.add_argument("--deeplab_confidence_threshold", default=0.99,
                        help="Logit threshold to keep or discard a sub-class")
    parser.add_argument("--deeplab_ready", default=False, type=bool,
                        help="Pretrained deeplab or not")
    return parser
