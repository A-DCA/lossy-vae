import argparse
import time
from pathlib import Path

# ...existing imports...

def save_checkpoint(model, optimizer, epoch, exp_dir, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    path = exp_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, path)
    logging.info(f"Saved checkpoint to {path}")

def train(config: ModelConfig):
    # Setup experiment directory
    exp_dir = config.get_exp_dir()
    logging.info(f"Experiment directory: {exp_dir}")
    
    # ...existing setup code...
    
    # Training loop
    for epoch in range(config.num_epochs):
        # ...existing epoch code...
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, exp_dir, epoch_loss)
        
        # ...existing logging code...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--exp-name', type=str, default='default',
                       help='Experiment name')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    args = parser.parse_args()
    
    config = ModelConfig()
    config.num_epochs = args.epochs
    config.exp_name = args.exp_name
    config.save_interval = args.save_interval
    
    train(config)
