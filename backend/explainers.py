"""Explainability module for AMSI-Net LULC model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from io import BytesIO
import PIL.Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

class ModelWrapper(nn.Module):
    """Wrapper to make AMSI-Net return only logits for explainability tools."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs['logits']

def array_to_base64(img_array):
    """Convert numpy RGB array to base64 jpeg string."""
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def get_saliency_map(model, image_tensor):
    """Generate vanilla Saliency map."""
    x = image_tensor.clone()
    x.requires_grad = True
    model.eval()
    
    # Forward pass
    outputs = model(x)
    logits = outputs['logits']
    
    # Get top class
    score, indices = torch.max(logits, 1)
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Saliency map = max absolute gradient across channels
    saliency, _ = torch.max(x.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    # Apply colormap
    saliency_heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    saliency_heatmap = cv2.cvtColor(saliency_heatmap, cv2.COLOR_BGR2RGB)
    saliency_heatmap = cv2.resize(saliency_heatmap, (224, 224))
    
    return array_to_base64(saliency_heatmap)

def get_gradcam_maps(model, image_tensor, orig_image_np):
    """Generate GradCAM and GradCAM++."""
    # Target the last convolutional layer of the ResNet backbone in AMSI-Net
    # In AMSI-Net, backbone.layer4 is the final block of the ResNet parts
    target_layers = [model.layer4[-1]]
    
    wrapper = ModelWrapper(model)
    
    # 1. GradCAM
    cam_algo = GradCAM(model=wrapper, target_layers=target_layers)
    grayscale_cam = cam_algo(input_tensor=image_tensor, targets=None)[0]
    
    orig_img_norm = orig_image_np.astype(np.float32) / 255.0
    cam_image = show_cam_on_image(orig_img_norm, grayscale_cam, use_rgb=True)
    
    # 2. GradCAM++
    cam_plus_algo = GradCAMPlusPlus(model=wrapper, target_layers=target_layers)
    grayscale_cam_plus = cam_plus_algo(input_tensor=image_tensor, targets=None)[0]
    cam_plus_image = show_cam_on_image(orig_img_norm, grayscale_cam_plus, use_rgb=True)
    
    return array_to_base64(cam_image), array_to_base64(cam_plus_image)

def get_lime_map(model, preprocessor, image_bytes, orig_image_np, device):
    """Generate LIME explanation."""
    explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(images):
        tensors = []
        for img_arr in images:
            # Re-apply preprocessing to each LIME perturbation
            img_pil = PIL.Image.fromarray(img_arr)
            tensor = preprocessor.transform(img_pil).to(device)
            tensors.append(tensor)
        
        batch_tensor = torch.stack(tensors)
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.sigmoid(outputs['logits'])  # Using sigmoid for multi-label
        return probs.cpu().numpy()
    
    explanation = explainer.explain_instance(
        orig_image_np, 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=100
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    img_boundary = mark_boundaries(temp / 255.0, mask)
    img_boundary_uint8 = np.uint8(img_boundary * 255)
    
    return array_to_base64(img_boundary_uint8)

def generate_all_explanations(model, preprocessor, image_tensor, image_bytes):
    """Generate all explainability maps."""
    device = image_tensor.device
    
    orig_pil = PIL.Image.open(BytesIO(image_bytes)).convert("RGB")
    orig_pil = orig_pil.resize((224, 224))
    orig_np = np.array(orig_pil)
    
    result = {}
    
    # Saliency
    try:
        saliency_b64 = get_saliency_map(model, image_tensor)
        result["Saliency"] = f"data:image/jpeg;base64,{saliency_b64}"
    except Exception as e:
        print(f"Error generating Saliency: {e}")
        
    # GradCAM
    try:
        gc_b64, gcp_b64 = get_gradcam_maps(model, image_tensor, orig_np)
        result["GradCAM"] = f"data:image/jpeg;base64,{gc_b64}"
        result["GradCAM++"] = f"data:image/jpeg;base64,{gcp_b64}"
    except Exception as e:
        print(f"Error generating GradCAM: {e}")
        
    # LIME
    try:
        lime_b64 = get_lime_map(model, preprocessor, image_bytes, orig_np, device)
        result["LIME"] = f"data:image/jpeg;base64,{lime_b64}"
    except Exception as e:
        print(f"Error generating LIME: {e}")
        
    return result
