#!/usr/bin/env python3
"""Demo training script to test the dashboard."""

import math
import random
import time
from pathlib import Path

# Add dashboard backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard" / "backend"))

from app.client import TrainingMonitor


def fake_training_loop() -> None:
    """Simulate a training loop with fake metrics."""
    
    # Initialize the training monitor
    monitor = TrainingMonitor(
        run_name="Demo Training Run",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "Adam",
            "model": "ResNet18",
        },
        dashboard_url="http://localhost:8000",
    )
    
    print(f"✓ Started training run: {monitor.run_id}")
    print("✓ Dashboard: http://localhost:3000")
    print()
    
    try:
        # Simulate 100 training steps
        for step in range(1, 101):
            # Simulate training metrics with realistic patterns
            # Loss decreases over time with some noise
            base_loss = 2.0 * math.exp(-step / 30) + 0.1
            loss = base_loss + random.gauss(0, 0.05)
            
            # Accuracy increases over time
            base_acc = 1.0 - math.exp(-step / 25)
            accuracy = base_acc + random.gauss(0, 0.02)
            accuracy = max(0, min(1, accuracy))  # Clamp to [0, 1]
            
            # Learning rate decay
            lr = 0.001 * (0.95 ** (step // 10))
            
            # Log metrics
            monitor.log_metrics({
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": lr,
            }, step=step)
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step:3d}: loss={loss:.4f}, accuracy={accuracy:.4f}, lr={lr:.6f}")
            
            # Save checkpoint every 25 steps
            if step % 25 == 0:
                checkpoint_path = f"./checkpoints/model_step_{step}.pt"
                monitor.save_checkpoint(
                    filepath=checkpoint_path,
                    metrics={"loss": loss, "accuracy": accuracy},
                )
                print(f"  → Saved checkpoint: {checkpoint_path}")
            
            # Simulate training time
            time.sleep(0.1)
        
        print()
        print("✓ Training completed successfully!")
        monitor.set_status("completed")
        
    except KeyboardInterrupt:
        print()
        print("⚠ Training interrupted by user")
        monitor.set_status("interrupted")
    
    except Exception as e:
        print()
        print(f"✗ Training failed: {e}")
        monitor.set_status("failed")
        raise


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Dashboard Demo - Training Script")
    print("=" * 60)
    print()
    print("This script simulates a training loop and sends metrics")
    print("to the dashboard in real-time.")
    print()
    print("Make sure the dashboard backend is running:")
    print("  cd dashboard/backend")
    print("  python -m app.main")
    print()
    print("=" * 60)
    print()
    
    input("Press Enter to start training...")
    print()
    
    fake_training_loop()


if __name__ == "__main__":
    main()


