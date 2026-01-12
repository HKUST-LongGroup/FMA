import datetime
import torch
import json
import os
import random
import sys
import numpy as np
import argparse
class SimpleLogger:
    """A simple logger that prints to both console and file."""
    def __init__(self, log_path='training.log'):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a")  # Use 'a' mode for appending

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def set_seed(seed):
    """
    Sets the seed for various random number generators to ensure reproducibility.
    """
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class DefaultConfig:
    def __init__(self):
        # =================================================================
        # Basic Training Parameters
        # =================================================================
        self.epochs = 600
        self.warmup_epochs = 1
        self.batch_size = 32
        self.lr = 2e-4
        self.weight_decay = 0.01
        self.optimizer = 'adamw'
        self.seed = 1
        self.gamma = 0 # control the stochastic noise level
        # =================================================================
        #  Model and Dataset Specific Parameters
        # =================================================================
        
        # ['ViT-B/16', 'RN50', 'RN101','ViT-B/32']
        self.clip_type = 'ViT-B/16' 
        # ['OxfordPets', 'EuroSAT', 'UCF101', 'SUN397', 'Caltech101',
        #  'DescribableTextures', 'FGVCAircraft', 'Food101', 'OxfordFlowers',
        #  'StanfordCars', 'ImageNet']
        self.dataset = 'EuroSAT' 

        self.num_shots = 16    # [1, 2, 4, 8, 16]
        self.subsample_classes = 'all'  # ['base', 'new', 'all']
        self.feature_extractor = 'coop'  # ['clip', 'coop','cocoop']
        self.blocks = 12

        # =================================================================
        # Other Parameters
        # =================================================================
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f'./checkpoints/{self.timestamp}'
        self.dataset_root = './data/'


    def save(self, file_path=None):
        """Save configuration to JSON file"""
        if file_path is None:
            # Ensure save directory exists
            os.makedirs(self.save_dir, exist_ok=True)
            file_path = os.path.join(self.save_dir, 'config.json')

        # Use __dict__ to get all instance attributes
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"✅ Config saved to {file_path}")

    @classmethod
    def from_json(cls, file_path):
        """Load configuration from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            attrs = json.load(f)
        
        # Create a default instance
        config = cls()
        # Update instance with loaded attributes
        config.__dict__.update(attrs)
        
        print(f"✅ Config loaded from {file_path}")
        return config
    
    def parse_args(self):
        """Parse command line arguments and update configuration"""
        parser = argparse.ArgumentParser(description='Train model with configurable parameters')
        
        parser.add_argument('--dataset', type=str, default=None,
                          help='Dataset name (e.g., EuroSAT, OxfordPets, UCF101, etc.)')
        parser.add_argument('--num_shots', type=int, default=None,
                          help='Number of shots for few-shot learning (e.g., 1, 2, 4, 8, 16)')
        parser.add_argument('--feature_extractor', type=str, default=None,
                          choices=['clip', 'coop', 'cocoop'],
                          help='Feature extractor to use (clip, coop, or cocoop)')
        parser.add_argument('--gamma', type=float, default=None,
                          help='Stochastic noise level for feature interpolation')
        args = parser.parse_args()
        
        # Only update user-specified parameters (non-None parameters)
        if args.dataset is not None:
            self.dataset = args.dataset
        if args.num_shots is not None:
            self.num_shots = args.num_shots
        if args.feature_extractor is not None:
            self.feature_extractor = args.feature_extractor
        if args.gamma is not None:
            self.gamma = args.gamma
        
        return self
    
    def __str__(self):
        """
        Return a formatted string to print all class configurations.
        """
        # Use self.__dict__ to get all instance attributes
        # .items() converts it to (key, value) pairs
        # Then format each line as "key: value"
        settings = "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])
        return f"===== Config Settings =====\n{settings}\n==========================="