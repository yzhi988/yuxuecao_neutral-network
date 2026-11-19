"""
Training script for PINN on MIT battery dataset

This script trains a Physics-Informed Neural Network for battery State of Health (SOH) prediction
using the MIT dataset. The model combines data-driven learning with physics constraints.
"""

from dataloader.dataloader import MITdata
from Model.Model import PINN
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    """Configure hyperparameters for MIT dataset"""
    parser = argparse.ArgumentParser(description='PINN Training on MIT Dataset')
    
    # Data configuration
    parser.add_argument('--data', type=str, default='MIT', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    parser.add_argument('--normalization_method', type=str, default='robust', 
                        choices=['min-max', 'z-score', 'robust'],
                        help='Data normalization method')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=500, help='Maximum training epochs')
    parser.add_argument('--early_stop', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr', type=float, default=2e-3, help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=5e-6, help='Final learning rate')
    parser.add_argument('--lr_F', type=float, default=2e-3, help='Learning rate for F network')
    
    # Model architecture
    parser.add_argument('--u_layers_num', type=int, default=5, help='Number of layers in u network')
    parser.add_argument('--u_hidden_dim', type=int, default=128, help='Hidden dimension of u network')
    parser.add_argument('--u_output_dim', type=int, default=5, help='Output dimension of u encoder')
    parser.add_argument('--u_dropout', type=float, default=0.2, help='Dropout rate for u network')
    parser.add_argument('--F_layers_num', type=int, default=3, help='Number of layers in F network')
    parser.add_argument('--F_hidden_dim', type=int, default=128, help='Hidden dimension of F network')
    parser.add_argument('--F_dropout', type=float, default=0.2, help='Dropout rate for F network')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0, 
                        help='Weight for PDE loss (L = L_data + alpha*L_PDE)')
    parser.add_argument('--beta', type=float, default=0.5, 
                        help='Weight for physics constraint loss')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='Log file name')
    parser.add_argument('--save_folder', type=str, default='results/MIT_results', 
                        help='Folder to save results')
    
    return parser.parse_args()


def load_MIT_data(args, small_sample=None):
    """
    Load MIT battery dataset
    
    Args:
        args: Argument namespace containing data configuration
        small_sample: If specified, limit training data to first N samples
        
    Returns:
        Dictionary containing train, validation, and test dataloaders
    """
    root = 'data/MIT data'
    train_list = []
    test_list = []
    
    # Split data by battery ID: every 5th battery for testing
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        for f in files:
            battery_id = int(f.split('-')[-1].split('.')[0])
            if battery_id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    
    # Optional: limit training data for small sample experiments
    if small_sample is not None:
        train_list = train_list[:small_sample]
    
    # Load data
    data = MITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    
    # Organize dataloaders
    dataloader = {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3']
    }
    
    return dataloader


def main():
    """Main training loop"""
    args = get_args()
    
    # Run multiple experiments
    num_experiments = 10
    for exp_id in range(num_experiments):
        # Set unique save folder for each experiment
        args.save_folder = f'results/MIT_results/Experiment_{exp_id + 1}'
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        
        print(f"\n{'='*60}")
        print(f"Starting Experiment {exp_id + 1}/{num_experiments}")
        print(f"{'='*60}\n")
        
        # Load data and train model
        dataloader = load_MIT_data(args)
        pinn = PINN(args)
        pinn.Train(
            trainloader=dataloader['train'],
            validloader=dataloader['valid'],
            testloader=dataloader['test']
        )
        
        print(f"\nExperiment {exp_id + 1} completed. Results saved to {args.save_folder}\n")


if __name__ == '__main__':
    main()
