import torch
from torchvision import transforms
from PIL import Image
import io

class Preprocessor:
    """Handles image preprocessing for LULC models."""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to a normalized tensor."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor

    def get_image_info(self, image_bytes: bytes) -> dict:
        """Extract metadata from image bytes."""
        image = Image.open(io.BytesIO(image_bytes))
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode
        }

def get_preprocessor():
    return Preprocessor()
