#!/usr/bin/env python3
"""
Alternative startup script for Tea Leaf Detection Website
This script sets up the environment to handle PyTorch compatibility issues.
"""

import os
import sys

# Set environment variable to handle PyTorch 2.6+ compatibility
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Import and run the main application
if __name__ == "__main__":
    try:
        # Import after setting environment
        import torch
        
        # Additional compatibility setup
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            try:
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.modules.Conv',
                    'ultralytics.nn.modules.Bottleneck',
                    'ultralytics.nn.modules.C2f',
                    'ultralytics.nn.modules.SPPF',
                    'ultralytics.nn.modules.Detect'
                ])
                print("✅ PyTorch safe globals configured for Ultralytics")
            except Exception as e:
                print(f"⚠️  Could not configure safe globals: {e}")
        
        # Import and run the main application
        from run import main
        main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)