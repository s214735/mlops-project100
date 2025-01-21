import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))
from p100.model import ResNetModel
from p100 import train
import unittest
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

def test_my_train():
    def setUp(self):
        """Set up mock configurations and resources for testing."""
        self.mock_config = {
            'data': {
                'processed_path': "./mock_dataset"
            },
            'train': {
                'batch_size': 4,
                'num_workers': 4,
                'lr': 0.001,
                'epochs': 1,
                'devices': 1,
                'log_every_n_steps': 10
            },
            'model': {
                'num_classes': 1000,
                'save_path': "./mock_model.pth"
            }
        }

        # Create a mock dataset directory if it doesn't exist
        if not os.path.exists(self.mock_config['data']['processed_path']):
            os.makedirs(self.mock_config['data']['processed_path'])
            
        # Generate mock data (e.g., random tensors saved as files)
        for i in range(10):
            torch.save(torch.randn(3, 28, 28), os.path.join(self.mock_config['data']['processed_path'], f'data_{i}.pt'))

    def tearDown(self):
        """Clean up resources after testing."""
        # Remove mock dataset and model file
        for file in os.listdir(self.mock_config['data']['processed_path']):
            os.remove(os.path.join(self.mock_config['data']['processed_path'], file))
        os.rmdir(self.mock_config['data']['processed_path'])

        if os.path.exists(self.mock_config['model']['save_path']):
            os.remove(self.mock_config['model']['save_path'])

    def test_training_process(self):
        """Test if the training process runs without errors for one epoch."""
        try:
            # Initialize model and dataloader
            model = ResNetModel(num_classes=self.mock_config['model']['num_classes'], lr=self.mock_config['train']['lr'])
            #dataloader = DataLoader(self.mock_config['dataset_path'], self.mock_config['batch_size'])
            dataloader = DataLoader(self.mock_config['processed_path'], batch_size=self.mock_config['train']['batch_size'], shuffle=False, num_workers=self.mock_config['train']['numworkers'] )

            # Train the model
            train.fit(model, dataloader,dataloader )

            # Assert that the model file is saved
            self.assertTrue(os.path.exists(self.mock_config['model_save_path']))
        except Exception as e:
            self.fail(f"Training process failed with exception: {e}")

if __name__ == '__main__':
    test_my_train()