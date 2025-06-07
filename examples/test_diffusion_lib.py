import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Add the diffusion_lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from diffusion_lib.method.ddpm import DDPM
from diffusion_lib.trainer.base_trainer import BaseTrainer
from diffusion_lib.sampler.base_sampler import BaseSampler
from diffusion_lib.logger.base_logger import BaseLogger
from diffusion_lib.evaluator.base_evaluator import BaseEvaluator
from diffusion_lib.utils.models import SimpleUNet2D


class TestDiffusionLib(unittest.TestCase):
    """Test suite for the diffusion library."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 4
        self.image_size = 32
        self.channels = 1
        self.num_timesteps = 100  # Reduced for faster testing
        
        # Create synthetic 2D data (simple geometric shapes)
        self.test_data = self._generate_synthetic_2d_data()
        self.dataloader = DataLoader(
            TensorDataset(self.test_data), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Initialize components
        self.diffusion_config = {
            'num_timesteps': self.num_timesteps,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'schedule_type': 'linear'
        }
        
        self.model = SimpleUNet2D(
            in_channels=self.channels,
            out_channels=self.channels,
            base_channels=32,  # Smaller for testing
            time_emb_dim=128,
            num_res_blocks=1,  # Reduced for faster testing
            use_attention=False,  # Disabled for faster testing
            dropout=0.1
        )
        
        self.method = DDPM(self.diffusion_config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.logger = BaseLogger(log_dir="test_logs", experiment_name="test_diffusion")
        self.evaluator = BaseEvaluator(device=self.device)
    
    def _generate_synthetic_2d_data(self) -> torch.Tensor:
        """
        Generate synthetic 2D data for testing.
        
        Returns:
            data: Synthetic 2D data tensor.
        """
        num_samples = 64
        data = torch.zeros(num_samples, self.channels, self.image_size, self.image_size)
        
        for i in range(num_samples):
            # Create different geometric shapes
            img = torch.zeros(self.image_size, self.image_size)
            
            if i % 4 == 0:
                # Circle
                center = (self.image_size // 2, self.image_size // 2)
                radius = self.image_size // 4
                y, x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
                mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
                img[mask] = 1.0
            elif i % 4 == 1:
                # Square
                start = self.image_size // 4
                end = 3 * self.image_size // 4
                img[start:end, start:end] = 1.0
            elif i % 4 == 2:
                # Horizontal line
                y_pos = self.image_size // 2
                img[y_pos-2:y_pos+2, :] = 1.0
            else:
                # Vertical line
                x_pos = self.image_size // 2
                img[:, x_pos-2:x_pos+2] = 1.0
            
            data[i, 0] = img
        
        # Normalize to [-1, 1]
        return data * 2.0 - 1.0
    
    def test_ddpm_forward_process(self):
        """Test DDPM forward diffusion process."""
        print("\n=== Testing DDPM Forward Process ===")
        
        # Test data
        x0 = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        x0 = x0.to(self.device)
        
        # Forward process
        xt, noise, t = self.method.forward_process(x0)
        
        # Assertions
        self.assertEqual(xt.shape, x0.shape, "Forward process should preserve shape")
        self.assertEqual(noise.shape, x0.shape, "Noise should have same shape as input")
        self.assertEqual(t.shape, (self.batch_size,), "Timesteps should match batch size")
        self.assertTrue(torch.all(t >= 0) and torch.all(t < self.num_timesteps), 
                       "Timesteps should be in valid range")
        
        print(f"✓ Forward process working correctly")
        print(f"  Input shape: {x0.shape}")
        print(f"  Noisy output shape: {xt.shape}")
        print(f"  Noise shape: {noise.shape}")
        print(f"  Timesteps shape: {t.shape}")
        print(f"  Timestep range: {t.min().item():.0f} - {t.max().item():.0f}")
    
    def test_ddpm_reverse_process(self):
        """Test DDPM reverse diffusion process."""
        print("\n=== Testing DDPM Reverse Process ===")
        
        # Set model in method
        self.method.set_model(self.model.to(self.device))
        
        # Test data
        xt = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        xt = xt.to(self.device)
        t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device)
        
        # Reverse process
        with torch.no_grad():
            x_prev = self.method.reverse_process(xt, t)
        
        # Assertions
        self.assertEqual(x_prev.shape, xt.shape, "Reverse process should preserve shape")
        self.assertTrue(torch.isfinite(x_prev).all(), "Output should be finite")
        
        print(f"✓ Reverse process working correctly")
        print(f"  Input shape: {xt.shape}")
        print(f"  Output shape: {x_prev.shape}")
        print(f"  Output range: {x_prev.min().item():.3f} - {x_prev.max().item():.3f}")
    
    def test_trainer(self):
        """Test the trainer functionality."""
        print("\n=== Testing Trainer ===")
        
        # Initialize trainer
        trainer = BaseTrainer(
            model=self.model,
            dataset=self.dataloader,
            method=self.method,
            optimizer=self.optimizer,
            logger=self.logger,
            evaluator=self.evaluator,
            device=self.device
        )
        
        # Test single training step
        initial_loss = None
        for batch in self.dataloader:
            loss = trainer._train_step(batch)
            initial_loss = loss
            break
        
        self.assertIsInstance(initial_loss, float, "Loss should be a float")
        self.assertTrue(initial_loss > 0, "Loss should be positive")
        self.assertTrue(np.isfinite(initial_loss), "Loss should be finite")
        
        print(f"✓ Single training step working correctly")
        print(f"  Initial loss: {initial_loss:.6f}")
        
        # Test short training run
        print("  Running 2 epochs of training...")
        trainer.train(num_epochs=2)
        
        print(f"✓ Training completed successfully")
    
    def test_sampler(self):
        """Test the sampler functionality."""
        print("\n=== Testing Sampler ===")
        
        # Initialize sampler
        sampler = BaseSampler(
            model=self.model,
            method=self.method,
            dataloader=self.dataloader,
            n_samples=2,  # Generate 2 samples per data point
            device=self.device
        )
        
        # Test conditional sampling
        print("  Testing conditional sampling...")
        samples = sampler.sample(num_batches=1)  # Sample from just 1 batch
        
        expected_shape = (self.batch_size, 2, self.channels, self.image_size, self.image_size)
        self.assertEqual(samples.shape, expected_shape, 
                        f"Samples should have shape {expected_shape}")
        self.assertTrue(torch.isfinite(samples).all(), "Samples should be finite")
        
        print(f"✓ Conditional sampling working correctly")
        print(f"  Generated samples shape: {samples.shape}")
        print(f"  Sample value range: {samples.min().item():.3f} - {samples.max().item():.3f}")
        
        # Test unconditional sampling
        print("  Testing unconditional sampling...")
        unconditional_samples = sampler.sample_unconditional(
            num_samples=4, 
            data_shape=(self.channels, self.image_size, self.image_size)
        )
        
        expected_shape = (4, self.channels, self.image_size, self.image_size)
        self.assertEqual(unconditional_samples.shape, expected_shape,
                        f"Unconditional samples should have shape {expected_shape}")
        
        print(f"✓ Unconditional sampling working correctly")
        print(f"  Generated samples shape: {unconditional_samples.shape}")
    
    def test_evaluator(self):
        """Test the evaluator functionality."""
        print("\n=== Testing Evaluator ===")
        
        # Create sample and reference data
        samples = torch.randn(4, self.channels, self.image_size, self.image_size)
        references = torch.randn(4, self.channels, self.image_size, self.image_size)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(samples, references)
        
        # Check required metrics exist
        required_metrics = ['mse', 'mae', 'psnr']
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Metric {metric} should be computed")
            self.assertIsInstance(metrics[metric], float, f"Metric {metric} should be float")
            self.assertTrue(np.isfinite(metrics[metric]), f"Metric {metric} should be finite")
        
        print(f"✓ Evaluator working correctly")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        # Test diversity metrics
        diversity_metrics = self.evaluator.compute_diversity_metrics(samples)
        print(f"✓ Diversity metrics computed")
        for metric, value in diversity_metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    def test_integration(self):
        """Test full integration: train a model and then sample from it."""
        print("\n=== Testing Full Integration ===")
        
        # Create a smaller model for faster testing
        small_model = SimpleUNet2D(
            in_channels=self.channels,
            out_channels=self.channels,
            base_channels=16,
            time_emb_dim=64,
            num_res_blocks=1,
            use_attention=False,
            dropout=0.0
        )
        
        small_method = DDPM({
            'num_timesteps': 50,  # Even smaller for integration test
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'schedule_type': 'linear'
        })
        
        small_optimizer = optim.Adam(small_model.parameters(), lr=1e-3)
        
        # Train for a few epochs
        print("  Training model...")
        trainer = BaseTrainer(
            model=small_model,
            dataset=self.dataloader,
            method=small_method,
            optimizer=small_optimizer,
            device=self.device
        )
        trainer.train(num_epochs=3)
        
        # Sample from trained model
        print("  Sampling from trained model...")
        sampler = BaseSampler(
            model=small_model,
            method=small_method,
            dataloader=self.dataloader,
            n_samples=1,
            device=self.device
        )
        
        samples = sampler.sample(num_batches=1)
        
        # Evaluate samples
        print("  Evaluating samples...")
        reference_batch = next(iter(self.dataloader)).to(self.device)
        samples_for_eval = samples.squeeze(1)  # Remove n_samples dimension
        
        metrics = self.evaluator.compute_metrics(samples_for_eval, reference_batch)
        
        print(f"✓ Full integration test completed successfully")
        print(f"  Final sample shape: {samples.shape}")
        print(f"  Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.6f}")
    
    def tearDown(self):
        """Clean up after tests."""
        # Close logger
        if hasattr(self, 'logger'):
            self.logger.close()
        
        # Clean up CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_basic_functionality_test():
    """Run a basic functionality test without unittest framework."""
    print("=" * 60)
    print("DIFFUSION LIBRARY BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test 1: Create and test DDPM
    print("\n1. Testing DDPM creation and basic operations...")
    ddpm_config = {
        'num_timesteps': 100,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'schedule_type': 'linear'
    }
    
    ddpm = DDPM(ddpm_config)
    x0 = torch.randn(2, 1, 16, 16).to(device)
    xt, noise, t = ddpm.forward_process(x0)
    print(f"   ✓ DDPM forward process: {x0.shape} -> {xt.shape}")
    
    # Test 2: Create and test model
    print("\n2. Testing UNet model...")
    model = SimpleUNet2D(in_channels=1, out_channels=1, base_channels=16, 
                        time_emb_dim=64, num_res_blocks=1, use_attention=False)
    model = model.to(device)
    ddpm.set_model(model)
    
    with torch.no_grad():
        x_prev = ddpm.reverse_process(xt, t)
    print(f"   ✓ DDPM reverse process: {xt.shape} -> {x_prev.shape}")
    
    # Test 3: Test training step
    print("\n3. Testing training step...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    predicted_noise = model(xt, t)
    loss = ddpm.compute_loss(predicted_noise, noise)
    loss.backward()
    optimizer.step()
    print(f"   ✓ Training step completed, loss: {loss.item():.6f}")
    
    # Test 4: Test sampling
    print("\n4. Testing sampling...")
    model.eval()
    with torch.no_grad():
        sample = torch.randn(1, 1, 16, 16).to(device)
        for i in reversed(range(0, 10, 2)):  # Sample every 2nd timestep for speed
            t_tensor = torch.full((1,), i, device=device, dtype=torch.long)
            sample = ddpm.reverse_process(sample, t_tensor)
    print(f"   ✓ Sampling completed: {sample.shape}")
    
    print("\n" + "=" * 60)
    print("ALL BASIC FUNCTIONALITY TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == '__main__':
    # Run basic functionality test first
    run_basic_functionality_test()
    
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE UNIT TESTS")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2) 