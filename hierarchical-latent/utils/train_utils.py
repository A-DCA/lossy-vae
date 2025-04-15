import torch
from pathlib import Path
import json
from datetime import datetime

class TrainingLogger:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_dir / f"training_log_{datetime.now():%Y%m%d_%H%M%S}.json"
        self.logs = []
        
    def log(self, metrics: dict):
        self.logs.append(metrics)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

def save_checkpoint(model, optimizer, epoch, loss, save_dir: Path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pt')
