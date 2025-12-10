"""
Inference Module
----------------
Contains classes for:
- Face Detection (Haar Cascade)
- Model Wrapping (Loading, Preprocessing, Prediction)
- Explainability (Grad-CAM visualization)
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import sys
import os
import yaml

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import EmotionModel

class FaceDetector:
    def __init__(self):
        # Load the pre-trained Haar Cascade classifier for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Failed to load Haar Cascade from {cascade_path}")

    def detect_faces(self, frame):
        """
        Detects faces in a BGR frame.
        Returns a list of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # Backward pass
        score = output[0, class_idx]
        score.backward()
        
        # Generate heatmap
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = torch.nn.functional.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0], output

class ModelWrapper:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        # Load Config to get class names and other params
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # We need to know the class names. 
        # Usually they are in the dataset, but we might not want to load the whole dataset here.
        # Let's assume standard folder structure or check if they are saved in checkpoint.
        # For now, I will scan the data directory to get class names, same as ImageFolder does.
        data_dir = os.path.join(os.path.dirname(config_path), '..', self.config['data']['data_dir'])
        self.classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        print(f"Classes: {self.classes}")

        # Initialize Model
        # Try to load with config's architecture first
        self.model = EmotionModel(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Load Checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Handle both full checkpoint dict and direct state_dict save
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            try:
                self.model.load_state_dict(state_dict)
                print(f"Loaded checkpoint from {checkpoint_path}")
                self.is_trained = True
            except RuntimeError as e:
                print(f"Error loading state dict with {self.config['model']['name']}: {e}")
                print("Attempting fallback architecture...")
                
                # Toggle architecture
                # Toggle architecture
                current_name = self.config['model'].get('name', 'resnet18')
                
                # Try all known architectures
                known_archs = ['resnet18', 'convnext_v2_nano', 'improved_cnn']
                fallback_success = False
                
                for arch in known_archs:
                    if arch == current_name: continue
                    
                    print(f"Switching to fallback: {arch}...")
                    fallback_config = self.config.copy()
                    fallback_config['model']['name'] = arch
                    
                    try:
                        self.model = EmotionModel(fallback_config)
                        self.model.to(self.device)
                        self.model.eval()
                        self.model.load_state_dict(state_dict)
                        print(f"Successfully loaded checkpoint with {arch}")
                        self.is_trained = True
                        self.config = fallback_config
                        fallback_success = True
                        break
                    except RuntimeError:
                        print(f"Failed to load with {arch}")
                
                if not fallback_success:
                    print(f"Failed to load checkpoint with any known architecture.")
                    self.is_trained = False
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
            self.is_trained = False

        # Transforms
        self.image_size = self.config['data']['image_size']
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize GradCAM
        # Determine target layer based on architecture
        target_layer = None
        inner_model = getattr(self.model, 'model', None)
        
        if inner_model:
            if hasattr(inner_model, 'layer4'): # ResNet
                target_layer = inner_model.layer4
            elif hasattr(inner_model, 'stages'): # ConvNeXt
                target_layer = inner_model.stages[-1]
            elif hasattr(inner_model, 'conv4'): # ImprovedCNN (last conv layer)
                target_layer = inner_model.conv4
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layer4'): # Fallback for old ResNet wrapper
             target_layer = self.model.backbone.layer4
             
        if target_layer:
            self.grad_cam = GradCAM(self.model, target_layer)
        else:
            print("Warning: Could not determine target layer for GradCAM. Visualization disabled.")
            self.grad_cam = None

    def predict(self, face_image_bgr):
        """
        Predicts emotion for a single face image (BGR numpy array).
        Returns: class_name, confidence, heatmap_overlay
        """
        # Convert BGR to RGB and PIL
        img_rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True # Needed for GradCAM backward
        
        # Inference & GradCAM
        if self.grad_cam:
            heatmap, output = self.grad_cam(input_tensor)
        else:
            # Just forward pass
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
            heatmap = None
        
        # Get prediction
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        class_name = self.classes[pred_idx.item()]
        confidence = conf.item()
        
        # Process Heatmap for Visualization
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, (face_image_bgr.shape[1], face_image_bgr.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Overlay heatmap on original image
            overlay = cv2.addWeighted(face_image_bgr, 0.6, heatmap, 0.4, 0)
        else:
            overlay = face_image_bgr.copy()
        
        return class_name, confidence, overlay
