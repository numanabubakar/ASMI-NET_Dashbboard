import torch
import os
from models_arch import AMSINet

class ModelLoader:
    """Utility to manage loading and managing the AMSI-Net model."""
    def __init__(self, model_path: str, num_labels: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.num_labels = num_labels
        self.model = None

    def load_model(self):
        """Load state_dict into the AMSI-Net architecture."""
        if self.model is not None:
            return self.model
            
        print(f"Loading model from {self.model_path} on {self.device}...")
        self.model = AMSINet(num_labels=self.num_labels, pretrained=False)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        # map_location handles CPU/GPU switching
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        # In case the state_dict is nested or saved differently
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")
        return self.model

    def get_device(self):
        return self.device

# Singleton-style accessor
_loader = None

def get_model_loader():
    global _loader
    if _loader is None:
        # Default path based on project structure
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(base_dir), "model", "best_model.pth")
        _loader = ModelLoader(model_path, num_labels=60)
    return _loader
