"""Verification script for AMSI-Net Model loading and architecture."""

import sys
import os
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models_arch import AMSINet
    from model_loader import get_model_loader
    from class_mappings import get_all_labels
    
    print("--- AMSI-Net Backend Verification ---")
    
    # 1. Check Labels
    labels = get_all_labels()
    print(f"✅ Class Mappings: Found {len(labels)} labels.")
    
    # 2. Check Device
    loader = get_model_loader()
    device = loader.get_device()
    print(f"ℹ️ Device: Using {device}")
    
    # 3. Load Model
    print("⏳ Loading model state_dict (this may take a moment)...")
    model = loader.load_model()
    print("✅ Model: Architecture initialized and weights loaded.")
    
    # 4. Dummy Inference
    print("🧪 Running dummy inference check...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        outputs = model(dummy_input)
        
    if 'logits' in outputs and outputs['logits'].shape == (1, 60):
        print("✅ Inference: Model produced correct output shape (1, 60).")
    else:
        print("❌ Inference: Unexpected output shape.")
        
    print("\n🚀 Everything looks good! You can now start the API with:")
    print("   python backend/app.py")

except ImportError as e:
    print(f"\n❌ Dependency Error: {e}")
    print("   Please run: pip install -r backend/requirements.txt")
except Exception as e:
    print(f"\n❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()
