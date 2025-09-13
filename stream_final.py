"""
🧠 OncoVista - Breast Cancer Classification Streamlit App
==========================================================

A professional Streamlit application for breast cancer patch classification using 
Multi-Expert EfficientNet with interactive mammogram analysis.

Features:
- Interactive mammogram viewing with zoom and pan
- ROI selection with fixed patch extraction (224x224)
- Real-time AI classification using Multi-Expert EfficientNet architecture
- Expert contribution analysis and gating visualization
- Professional-grade UI with confidence scoring
- Support for DICOM, PNG, JPEG, and URL uploads

Author: AI Medical Imaging Team
Date: August 2025
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b1
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw
import requests
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import base64
import json
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.cm as cm

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="🔬 OncoVista",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Model configuration
NUM_CLASSES = 3
NUM_EXPERTS = 4
CLASS_NAMES = ['benign', 'malignant', 'normal']
IMG_SIZE = 224
PATCH_SIZE = 224

# Color coding for results
CLASS_COLORS = {
    'normal': '#28a745',    # Green
    'benign': '#ffc107',    # Yellow/Amber
    'malignant': '#dc3545'  # Red
}

CLASS_EMOJIS = {
    'normal': '✅',
    'benign': '⚠️',
    'malignant': '🔴'
}

# Device configuration
@st.cache_resource
def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()

# =============================================================================
# MULTI-EXPERT EFFICIENTNET MODEL ARCHITECTURE (EXACT REPLICA FROM NOTEBOOK)
# =============================================================================

class ExpertNetwork(nn.Module):
    """Individual expert network with specialized knowledge"""
    
    def __init__(self, in_features, num_classes, expert_id, dropout_rate=0.3):
        super(ExpertNetwork, self).__init__()
        
        self.expert_id = expert_id
        
        # Each expert has a unique architecture for diverse learning
        if expert_id == 0:
            # Expert 0: Deep narrow network (focused learning)
            self.layers = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, num_classes)
            )
        elif expert_id == 1:
            # Expert 1: Wide shallow network (broad pattern recognition)
            self.layers = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        elif expert_id == 2:
            # Expert 2: Residual-style network
            self.fc1 = nn.Linear(in_features, 384)
            self.bn1 = nn.BatchNorm1d(384)
            self.fc2 = nn.Linear(384, 384)
            self.bn2 = nn.BatchNorm1d(384)
            self.fc3 = nn.Linear(384, num_classes)
            self.dropout = nn.Dropout(dropout_rate)
            self.relu = nn.ReLU(inplace=True)
        else:
            # Expert 3: Enhanced dense network with channel attention
            self.layers = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.7),
                nn.Linear(256, num_classes)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.expert_id in [0, 1, 3]:
            return self.layers(x)
        elif self.expert_id == 2:
            # Residual connection
            identity = x
            out = self.fc1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            
            out = self.fc2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.dropout(out)
            
            # Add residual connection if dimensions match
            if identity.shape[1] == out.shape[1]:
                out = out + identity
            
            return self.fc3(out)

class GatingNetwork(nn.Module):
    """Gating network that determines which experts to use for each input"""
    
    def __init__(self, in_features, num_experts, dropout_rate=0.2):
        super(GatingNetwork, self).__init__()
        
        self.num_experts = num_experts
        
        # Feature processing for gating decision
        self.feature_processor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Gating decision layers
        self.gate = nn.Sequential(
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
        
        # Confidence estimation
        self.confidence = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Process features
        processed = self.feature_processor(x)
        
        # Generate gating weights
        gate_weights = self.gate(processed)
        
        # Generate confidence scores
        confidence = self.confidence(processed)
        
        return gate_weights, confidence

class MoEffNetClassifier(nn.Module):
    """Multi-Expert EfficientNet for advanced patch classification"""
    
    def __init__(self, num_classes=NUM_CLASSES, num_experts=NUM_EXPERTS, pretrained=True):
        super(MoEffNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        # EfficientNet-B1 backbone
        self.backbone = efficientnet_b1(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier and get features before pooling
        self.backbone_features = self.backbone.features
        self.backbone_avgpool = self.backbone.avgpool
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(in_features, num_classes, expert_id=i)
            for i in range(num_experts)
        ])
        
        # Gating network for expert selection
        self.gating_network = GatingNetwork(in_features, num_experts)
        
        # Feature enhancement layer
        self.feature_enhancer = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Extract features with EfficientNet-B1 backbone
        features = self.backbone_features(x)
        
        # Global average pooling
        features = self.global_avg_pool(features)
        features = torch.flatten(features, 1)
        
        # Enhance features
        enhanced_features = self.feature_enhancer(features)
        
        # Get gating weights and confidence
        gate_weights, confidence = self.gating_network(enhanced_features)
        
        # Get predictions from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(enhanced_features)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, num_classes]
        
        # Apply gating weights
        gate_weights = gate_weights.unsqueeze(2)  # [batch, num_experts, 1]
        
        # Weighted combination of expert outputs
        final_output = torch.sum(expert_outputs * gate_weights, dim=1)
        
        # Apply confidence weighting
        final_output = final_output * confidence
        
        return final_output
    
    def get_expert_analysis(self, x):
        """Get detailed expert analysis for visualization"""
        self.eval()
        with torch.no_grad():
            # Extract features
            features = self.backbone_features(x)
            features = self.global_avg_pool(features)
            features = torch.flatten(features, 1)
            enhanced_features = self.feature_enhancer(features)
            
            # Get gating info
            gate_weights, confidence = self.gating_network(enhanced_features)
            
            # Get individual expert outputs
            expert_outputs = []
            expert_probs = []
            for expert in self.experts:
                expert_output = expert(enhanced_features)
                expert_prob = F.softmax(expert_output, dim=1)
                expert_outputs.append(expert_output)
                expert_probs.append(expert_prob)
            
            # Final ensemble output
            expert_outputs_stacked = torch.stack(expert_outputs, dim=1)
            gate_weights_expanded = gate_weights.unsqueeze(2)
            final_output = torch.sum(expert_outputs_stacked * gate_weights_expanded, dim=1)
            final_output = final_output * confidence
            final_probs = F.softmax(final_output, dim=1)
            
            return {
                'final_probs': final_probs.cpu().numpy(),
                'expert_probs': [ep.cpu().numpy() for ep in expert_probs],
                'gate_weights': gate_weights.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'final_output': final_output.cpu().numpy()
            }

# =============================================================================
# MODEL LOADING AND CACHING
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained Multi-Expert EfficientNet model"""
    model_path = "best_patch_classifier.pth"
    
    try:
        # Create model instance
        model = MoEffNetClassifier(num_classes=NUM_CLASSES, num_experts=NUM_EXPERTS, pretrained=False)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, None
        
    except Exception as e:
        return None, str(e)

# =============================================================================
# IMAGE PROCESSING AND TRANSFORMS
# =============================================================================

@st.cache_data
def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet standards
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert('RGB')
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# =============================================================================
# RECTANGLE SELECTION SYSTEM - STREAMLIT IMAGE COORDINATES
# =============================================================================

def resize_image_for_display(image, max_height=600):
    """
    Resize image for display with fixed maximum height while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_height: Maximum height in pixels for display
    
    Returns:
        tuple: (resized_image, scale_factor)
    """
    original_width, original_height = image.size
    
    # If image height is less than max_height, don't resize
    if original_height <= max_height:
        return image, 1.0
    
    # Calculate scale factor based on height constraint
    scale_factor = max_height / original_height
    new_width = int(original_width * scale_factor)
    new_height = max_height
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image, scale_factor

def scale_coordinates_to_original(coordinates, scale_factor):
    """
    Scale coordinates from display image back to original image size
    
    Args:
        coordinates: Tuple of ((x1, y1), (x2, y2)) from display image
        scale_factor: Scale factor used for display
    
    Returns:
        Scaled coordinates for original image
    """
    if not coordinates or scale_factor == 1.0:
        return coordinates
    
    point1, point2 = coordinates
    x1, y1 = point1
    x2, y2 = point2
    
    # Scale back to original size
    orig_x1 = int(x1 / scale_factor)
    orig_y1 = int(y1 / scale_factor)
    orig_x2 = int(x2 / scale_factor)
    orig_y2 = int(y2 / scale_factor)
    
    return ((orig_x1, orig_y1), (orig_x2, orig_y2))

def get_rectangle_coords(coordinates):
    """Convert streamlit_image_coordinates format to standard rectangle coords"""
    if coordinates and len(coordinates) == 2:
        point1, point2 = coordinates
        x1 = min(point1[0], point2[0])
        y1 = min(point1[1], point2[1])
        x2 = max(point1[0], point2[0])
        y2 = max(point1[1], point2[1])
        return (x1, y1, x2, y2)
    return None

def create_rectangle_overlay(image, coordinates):
    """Draw rectangle overlay on image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    if coordinates:
        coords = get_rectangle_coords(coordinates)
        if coords:
            draw.rectangle(coords, fill=None, outline="red", width=3)
            
            # Add corner markers for visual feedback
            x1, y1, x2, y2 = coords
            marker_size = 8
            
            # Top-left corner
            draw.rectangle((x1-marker_size//2, y1-marker_size//2, 
                          x1+marker_size//2, y1+marker_size//2), 
                         fill="red", outline="red")
            
            # Top-right corner
            draw.rectangle((x2-marker_size//2, y1-marker_size//2, 
                          x2+marker_size//2, y1+marker_size//2), 
                         fill="red", outline="red")
            
            # Bottom-left corner
            draw.rectangle((x1-marker_size//2, y2-marker_size//2, 
                          x1+marker_size//2, y2+marker_size//2), 
                         fill="red", outline="red")
            
            # Bottom-right corner
            draw.rectangle((x2-marker_size//2, y2-marker_size//2, 
                          x2+marker_size//2, y2+marker_size//2), 
                         fill="red", outline="red")
    
    return img

def extract_rectangle_patch(image, coordinates):
    """Extract and resize patch from rectangle coordinates"""
    if not coordinates:
        return None
        
    coords = get_rectangle_coords(coordinates)
    if not coords:
        return None
    
    x1, y1, x2, y2 = coords
    
    # Check if whole image is selected
    image_width, image_height = image.size
    is_whole_image = (x1 == 0 and y1 == 0 and x2 == image_width and y2 == image_height)
    
    if not is_whole_image:
        # Ensure we have a reasonable rectangle size for manual selection
        if abs(x2 - x1) < 20 or abs(y2 - y1) < 20:
            return None
    
    # Extract the rectangle from image
    patch = image.crop(coords)
    
    # Resize to standard patch size for model input
    patch_resized = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    
    return patch_resized, patch, coords

def create_rectangle_selection_interface(image):
    """Create the rectangle selection interface with constrained height display"""
    
    # Initialize coordinates and scale factor in session state
    if "coordinates" not in st.session_state:
        st.session_state["coordinates"] = None
    if "display_scale_factor" not in st.session_state:
        st.session_state["display_scale_factor"] = 1.0
    
    # Get max display height from settings (default 600 if not set)
    max_height = st.session_state.get("max_display_height", 600)
    
    # Create columns for layout
    col_image, col_patch, col_controls = st.columns([3, 1, 1])
    
    with col_image:
        st.subheader("🖼️ Select Region of Interest")
        
        # ROI selection options
        roi_method = st.radio(
            "Choose ROI selection method:",
            ["🎯 Manual Rectangle", "🖼️ Whole Image"],
            horizontal=True,
            help="Select either a custom rectangle or use the entire image"
        )
        
        if roi_method == "🖼️ Whole Image":
            st.markdown("**Whole image selected for analysis**")
            # Set coordinates to cover the entire image
            image_width, image_height = image.size
            st.session_state["coordinates"] = ((0, 0), (image_width, image_height))
            st.session_state["display_scale_factor"] = 1.0
            # Show the whole image with a border to indicate selection
            img_with_overlay = create_rectangle_overlay(image, st.session_state["coordinates"])
            
            # Resize for display
            display_image, scale_factor = resize_image_for_display(img_with_overlay, max_height=max_height)
            st.image(display_image, caption="Entire image selected for analysis")
            
        else:
            st.markdown("**Instructions:** Click and drag to select a rectangular region for analysis")
            
            # Resize image for display to constrain height
            display_image, scale_factor = resize_image_for_display(image, max_height=max_height)
            st.session_state["display_scale_factor"] = scale_factor
            
            # Create overlay on display image (scale coordinates for display)
            display_coordinates = st.session_state["coordinates"]
            if display_coordinates and scale_factor != 1.0:
                # Scale coordinates for display
                point1, point2 = display_coordinates
                x1, y1 = point1
                x2, y2 = point2
                
                # Scale coordinates to display size
                display_x1 = int(x1 * scale_factor)
                display_y1 = int(y1 * scale_factor)
                display_x2 = int(x2 * scale_factor)
                display_y2 = int(y2 * scale_factor)
                
                display_coordinates = ((display_x1, display_y1), (display_x2, display_y2))
            
            img_with_overlay = create_rectangle_overlay(display_image, display_coordinates)
            
            # Show image dimensions info
            orig_w, orig_h = image.size
            disp_w, disp_h = display_image.size
            if scale_factor != 1.0:
                st.caption(f"**Display:** {disp_w}×{disp_h} (scaled from {orig_w}×{orig_h}, factor: {scale_factor:.2f})")
            else:
                st.caption(f"**Size:** {orig_w}×{orig_h}")
            
            # Image coordinates selector
            value = streamlit_image_coordinates(
                img_with_overlay, 
                key="rectangle_selector", 
                click_and_drag=True
            )
            
            # Handle rectangle selection
            if value is not None:
                # Get coordinates from display image
                display_point1 = value["x1"], value["y1"]
                display_point2 = value["x2"], value["y2"]
                
                # Scale back to original image coordinates
                if scale_factor != 1.0:
                    orig_x1 = int(display_point1[0] / scale_factor)
                    orig_y1 = int(display_point1[1] / scale_factor)
                    orig_x2 = int(display_point2[0] / scale_factor)
                    orig_y2 = int(display_point2[1] / scale_factor)
                    original_coords = ((orig_x1, orig_y1), (orig_x2, orig_y2))
                else:
                    original_coords = (display_point1, display_point2)
                
                # Only update if we have a valid rectangle and it's different from current
                if (display_point1[0] != display_point2[0] and 
                    display_point1[1] != display_point2[1] and 
                    st.session_state["coordinates"] != original_coords):
                    
                    st.session_state["coordinates"] = original_coords
                    st.rerun()
    
    with col_patch:
        st.subheader("� Selected Region")
        
        # Show enlarged selected region
        if st.session_state["coordinates"]:
            result = extract_rectangle_patch(image, st.session_state["coordinates"])
            
            if result:
                patch_resized, patch_original, coords = result
                
                # Check if whole image is selected
                image_width, image_height = image.size
                is_whole_image = (coords[0] == 0 and coords[1] == 0 and 
                                coords[2] == image_width and coords[3] == image_height)
                
                if is_whole_image:
                    # Show the resized whole image
                    st.image(patch_resized, caption="Whole Image (Resized to 224x224)", use_column_width=True)
                    st.caption(f"**Original Size:** {image_width} × {image_height} pixels")
                    st.caption(f"**Resized to:** {PATCH_SIZE} × {PATCH_SIZE} pixels for analysis")
                else:
                    # Show original selection (enlarged)
                    enlargement_factor = 1.5
                    enlarged_patch = patch_original.resize(
                        (int(patch_original.width * enlargement_factor), 
                         int(patch_original.height * enlargement_factor)),
                        Image.Resampling.LANCZOS
                    )
                    
                    st.image(enlarged_patch, caption="Selected Region (Enlarged)", use_column_width=True)
                    
                    # Show coordinates info
                    x1, y1, x2, y2 = coords
                    st.caption(f"**Region:** ({x1}, {y1}) to ({x2}, {y2})")
                    st.caption(f"**Size:** {x2-x1} × {y2-y1} pixels")
                
                # Store for analysis
                st.session_state["current_patch"] = patch_resized
                st.session_state["patch_coords"] = coords
                st.session_state["original_patch"] = patch_original
                
            else:
                st.info("👆 Draw a larger rectangle")
        else:
            st.info("👈 Select a region on the image")
    
    with col_controls:
        st.subheader("🎛️ Controls")
        
        # Analysis button
        analyze_button = st.button(
            "🔬 Analyze Region", 
            help="Analyze the selected region with Multi-Expert EfficientNet",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.get("coordinates")
        )
        
        # Clear selection
        if st.button("🔄 Clear Selection", use_container_width=True):
            st.session_state["coordinates"] = None
            st.session_state["display_scale_factor"] = 1.0
            if "current_patch" in st.session_state:
                del st.session_state["current_patch"]
            if "patch_coords" in st.session_state:
                del st.session_state["patch_coords"]
            st.rerun()
        
        # Quick selection helpers
        st.markdown("---")
        st.subheader("📏 Selection Info")
        
        # Show scale factor info
        scale_factor = st.session_state.get("display_scale_factor", 1.0)
        if scale_factor != 1.0:
            st.caption(f"**Display Scale:** {scale_factor:.2f}x (height constrained to 600px)")
        
        if st.session_state["coordinates"]:
            coords = get_rectangle_coords(st.session_state["coordinates"])
            if coords:
                x1, y1, x2, y2 = coords
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                st.metric("Width", f"{width}px")
                st.metric("Height", f"{height}px")
                st.metric("Area", f"{area:,}px²")
                
                # Check if whole image is selected
                image_width, image_height = image.size
                is_whole_image = (x1 == 0 and y1 == 0 and x2 == image_width and y2 == image_height)
                
                if is_whole_image:
                    st.success("🖼️ Whole image selected for analysis!")
                elif width < 100 or height < 100:
                    st.warning("⚠️ Small region selected. Consider selecting a larger area for better analysis.")
                elif width > 800 or height > 800:
                    st.info("ℹ️ Large region selected. Analysis will focus on key features.")
                else:
                    st.success("✅ Good region size for analysis!")
        
        return analyze_button
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size
    
    # Adjust if bounds exceed image dimensions
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_width:
        x1 -= (x2 - image_width)
        x2 = image_width
    if y2 > image_height:
        y1 -= (y2 - image_height)
        y2 = image_height
    
    # Ensure minimum size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, max(x1 + 50, x2))
    y2 = min(image_height, max(y1 + 50, y2))
    
    return (x1, y1, x2, y2)

# =============================================================================
# EXPLAINABLE AI (XAI) - GRADCAM AND ATTENTION VISUALIZATION
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Multi-Expert EfficientNet interpretability.
    Adapted for multi-expert architecture.
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._get_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _get_target_layer(self):
        """Get the last convolutional layer of the EfficientNet backbone"""
        # For Multi-Expert EfficientNet with EfficientNet-B1, get the last conv layer from features
        if hasattr(self.model, 'backbone'):
            # Handle DataParallel wrapper
            backbone = self.model.backbone if not isinstance(self.model, nn.DataParallel) else self.model.module.backbone
            
            # EfficientNet structure: features -> classifier
            if hasattr(backbone, 'features'):
                layers = list(backbone.features.modules())
            else:
                layers = list(backbone.modules())
            
            # Find the last Conv2d layer
            for layer in reversed(layers):
                if isinstance(layer, nn.Conv2d):
                    return layer
        return None
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        if self.target_layer:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Class Activation Map for Multi-Expert EfficientNet
        
        Args:
            input_tensor: Input image tensor
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            numpy array: Heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target = output[:, class_idx]
        target.backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            return self._generate_feature_cam(input_tensor)
        
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.numpy()
    
    def _generate_feature_cam(self, input_tensor):
        """Fallback method using feature maps directly"""
        # Get feature maps from the backbone
        model_to_use = self.model if not isinstance(self.model, nn.DataParallel) else self.model.module
        
        with torch.no_grad():
            features = model_to_use.backbone_features(input_tensor)
            
        # Average across channels to create attention map
        attention = torch.mean(features, dim=1, keepdim=True)
        attention = F.interpolate(attention, size=(224, 224), mode='bilinear', align_corners=False)
        attention = attention.squeeze().cpu().numpy()
        
        # Normalize to 0-1
        if attention.max() > attention.min():
            attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        return attention

def create_attention_heatmap(model, input_tensor, target_size=(224, 224)):
    """
    Create attention heatmap using feature maps from Multi-Expert EfficientNet backbone
    """
    model.eval()
    
    # Handle DataParallel wrapper
    model_to_use = model if not isinstance(model, nn.DataParallel) else model.module
    
    # Get feature maps from the backbone
    with torch.no_grad():
        features = model_to_use.backbone_features(input_tensor)
        
    # Average across channels to create attention map
    attention = torch.mean(features, dim=1, keepdim=True)
    attention = F.interpolate(attention, size=target_size, mode='bilinear', align_corners=False)
    attention = attention.squeeze().cpu().numpy()
    
    # Normalize to 0-1
    if attention.max() > attention.min():
        attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    return attention

def overlay_heatmap_on_image(image, heatmap, alpha=0.6, colormap_name='jet'):
    """
    Overlay heatmap on original image using matplotlib colormap
    
    Args:
        image: Original image (PIL or numpy)
        heatmap: Heatmap array
        alpha: Transparency factor
        colormap_name: matplotlib colormap name
    
    Returns:
        PIL Image: Overlayed image
    """
    import matplotlib.cm as cm
    
    # Convert image to numpy if PIL
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Resize heatmap to image size
    if heatmap.shape != image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    # Get colormap
    colormap = cm.get_cmap(colormap_name)
    
    # Apply colormap to heatmap
    heatmap_colored = colormap(heatmap)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Overlay
    overlayed = cv2.addWeighted(image_np, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlayed)

# =============================================================================
# INTERACTIVE IMAGE VIEWER WITH ROI SELECTION
# =============================================================================

# This old function is no longer needed - using create_interactive_image_viewer instead
    
    return fig, roi_bounds

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_comprehensive_xai_analysis(analysis_results, patch_image, gradcam_heatmap, attention_heatmap):
    """Create comprehensive XAI analysis visualization"""
    # Create subplot layout for XAI analysis
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Original Patch', 'GradCAM Heatmap', 'Feature Attention',
            'Expert Predictions', 'Gating Analysis', 'XAI Summary'
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "xy"}]
        ]
    )
    
    # Convert images for display
    patch_array = np.array(patch_image)
    
    # Row 1: Images
    # Original patch
    fig.add_trace(
        go.Image(z=patch_array, name="Original"),
        row=1, col=1
    )
    
    # GradCAM overlay
    gradcam_overlay = overlay_heatmap_on_image(patch_image, gradcam_heatmap, alpha=0.5)
    gradcam_array = np.array(gradcam_overlay)
    fig.add_trace(
        go.Image(z=gradcam_array, name="GradCAM"),
        row=1, col=2
    )
    
    # Attention overlay
    attention_overlay = overlay_heatmap_on_image(patch_image, attention_heatmap, alpha=0.5, colormap_name='plasma')
    attention_array = np.array(attention_overlay)
    fig.add_trace(
        go.Image(z=attention_array, name="Attention"),
        row=1, col=3
    )
    
    # Row 2: Analysis charts
    expert_probs = analysis_results['expert_probs']
    gate_weights = analysis_results['gate_weights'][0]
    final_probs = analysis_results['final_probs'][0]
    
    # Expert predictions
    expert_names = [f'Expert {i+1}' for i in range(NUM_EXPERTS)]
    for i, class_name in enumerate(CLASS_NAMES):
        expert_class_probs = [expert_probs[j][0][i] for j in range(NUM_EXPERTS)]
        fig.add_trace(
            go.Bar(
                x=expert_names,
                y=expert_class_probs,
                name=f'{class_name.title()}',
                marker_color=CLASS_COLORS[class_name],
                opacity=0.8,
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Gating weights
    fig.add_trace(
        go.Bar(
            x=expert_names,
            y=gate_weights,
            name='Gate Weight',
            marker_color='lightblue',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # XAI Summary - key metrics
    xai_metrics = ['GradCAM Focus', 'Attention Spread', 'Expert Consensus', 'Model Confidence']
    xai_values = [
        np.max(gradcam_heatmap),  # Peak activation
        np.std(attention_heatmap),  # Attention spread
        len(set(np.argmax([ep[0] for ep in expert_probs], axis=1))) / NUM_EXPERTS,  # Consensus
        analysis_results['confidence'][0][0]  # Confidence
    ]
    
    fig.add_trace(
        go.Bar(
            x=xai_metrics,
            y=xai_values,
            name='XAI Metrics',
            marker_color=['red', 'blue', 'green', 'purple'],
            showlegend=False
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="🔍 Comprehensive XAI Analysis - Multi-Expert EfficientNet Explainability",
        title_x=0.5,
        showlegend=True
    )
    
    # Remove axes for image plots
    for row in [1]:
        for col in [1, 2, 3]:
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col)
    
    return fig

def create_expert_analysis_plot(analysis_results):
    """Create comprehensive expert analysis visualization"""
    expert_probs = analysis_results['expert_probs']
    gate_weights = analysis_results['gate_weights'][0]  # First sample
    final_probs = analysis_results['final_probs'][0]    # First sample
    confidence = analysis_results['confidence'][0][0]   # First sample
    
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Expert Predictions', 'Gating Weights', 
            'Final Ensemble Result', 'Expert Contributions'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Expert predictions heatmap-style bar chart
    expert_names = [f'Expert {i+1}' for i in range(NUM_EXPERTS)]
    
    for i, class_name in enumerate(CLASS_NAMES):
        expert_class_probs = [expert_probs[j][0][i] for j in range(NUM_EXPERTS)]
        fig.add_trace(
            go.Bar(
                x=expert_names,
                y=expert_class_probs,
                name=f'{class_name.title()}',
                marker_color=CLASS_COLORS[class_name],
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Gating weights
    fig.add_trace(
        go.Bar(
            x=expert_names,
            y=gate_weights,
            name='Gate Weight',
            marker_color='lightblue',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Final ensemble result
    fig.add_trace(
        go.Bar(
            x=CLASS_NAMES,
            y=final_probs,
            name='Final Prediction',
            marker_color=[CLASS_COLORS[name] for name in CLASS_NAMES],
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Expert contributions (pie chart)
    fig.add_trace(
        go.Pie(
            labels=expert_names,
            values=gate_weights,
            name="Expert Contributions",
            showlegend=False,
            textinfo='label+percent',
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text=f"🧠 Multi-Expert EfficientNet Expert Analysis (Confidence: {confidence:.1%})",
        title_x=0.5,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Experts", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_xaxes(title_text="Experts", row=1, col=2)
    fig.update_yaxes(title_text="Gate Weight", row=1, col=2)
    fig.update_xaxes(title_text="Classes", row=2, col=1)
    fig.update_yaxes(title_text="Final Probability", row=2, col=1)
    
    return fig

def create_confidence_gauge(confidence):
    """Create a confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Model Confidence"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# =============================================================================
# STREAMLIT APP INTERFACE
# =============================================================================

def main():

    
    # Main header
    st.title("🔬 OncoVista")
    st.write("Multi-Expert EfficientNet - Interactive Mammogram Analysis Platform")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Phase information
        st.info("""
        🧪 Best Practice Pipeline (Phased)
        
        **✅ Phase 1:** Train patch classifier
        
        **🔁 Phase 2:** Evaluate on manual ROI extractions
        
        **🚧 Phase 3:** See XAI Visualiszation
        
        **🧠 Phase 4:** End-to-end diagnostic pipeline
        """)
        
        st.subheader("📊 Model Information")
        st.info("""
        **Architecture:** Multi-Expert EfficientNet-B1  
        **Experts:** 4 specialized networks  
        **Classes:** Normal, Benign, Malignant  
        **Input Size:** 224×224 pixels  
        **Backbone:** EfficientNet-B1 (ImageNet pretrained)
        """)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            show_expert_analysis = st.checkbox("Show Expert Analysis", value=True)
            show_confidence_gauge = st.checkbox("Show Confidence Gauge", value=True)
            patch_overlay_opacity = st.slider("Patch Overlay Opacity", 0.0, 1.0, 0.3)
            
            # Image display settings
            st.markdown("**Image Display Settings**")
            max_display_height = st.slider(
                "Max Display Height (px)", 
                min_value=400, 
                max_value=1000, 
                value=600, 
                step=50,
                help="Maximum height for image display. Width adjusts proportionally."
            )
            st.session_state["max_display_height"] = max_display_height
    
    # Load model
    with st.spinner("🔄 Loading Multi-Expert EfficientNet model..."):
        model, error = load_model()
    
    if model is None:
        st.error(f"❌ Failed to load model: {error}")
        st.info("Please ensure 'best_patch_classifier.pth' is in the current directory.")
        return
    
    st.success("✅ Multi-Expert EfficientNet model loaded successfully!")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "📖 About Multi-Expert EfficientNet"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("1️⃣ Upload Mammogram")
            
            # Upload options
            upload_method = st.radio(
                "Choose upload method:",
                ["📁 File Upload", "🌐 URL"],
                horizontal=True
            )
            
            uploaded_image = None
            
            if upload_method == "📁 File Upload":
                uploaded_file = st.file_uploader(
                    "Choose a mammogram image...",
                    type=['png', 'jpg', 'jpeg', 'dcm'],
                    help="Supports PNG, JPEG, and DICOM files"
                )
                if uploaded_file is not None:
                    uploaded_image = Image.open(uploaded_file).convert('RGB')
            
            elif upload_method == "🌐 URL":
                image_url = st.text_input(
                    "Enter image URL:",
                    placeholder="https://example.com/mammogram.jpg"
                )
                if image_url:
                    uploaded_image = load_image_from_url(image_url)
        
        with col2:
            st.subheader("📋 Instructions")
            st.markdown("""
            **How to use:**
            1. **Upload** a mammogram image
            2. **Choose ROI selection:** Either draw a rectangle around suspicious areas OR select the whole image
            3. **Click "Analyze Region"** to get AI classification
            4. **View detailed results** and expert analysis
            
            **Tips:**
            - Use "Manual Rectangle" for specific suspicious areas
            - Use "Whole Image" for overall image analysis
            - Larger regions provide more context
            - Check confidence scores and expert analysis
            - Use the dashboard for detailed insights
            """)
        
        # Image analysis section
        if uploaded_image is not None:
            # Store image in session state
            st.session_state.current_image = uploaded_image
            
            st.subheader("2️⃣ Interactive ROI Selection")
            st.markdown("**Choose your analysis method:** Select a specific region or analyze the entire image.")
            
            # Create rectangle selection interface
            analyze_roi = create_rectangle_selection_interface(uploaded_image)
            
            # Analysis results section
            if analyze_roi and st.session_state.get("current_patch") is not None:
                st.subheader("� Analysis Results")
                
                col_results1, col_results2 = st.columns([1, 1])
                
                with col_results1:
                    # Show processed patch (model input)
                    st.subheader("📋 Model Input")
                    st.image(
                        st.session_state["current_patch"], 
                        caption=f"Resized to {PATCH_SIZE}×{PATCH_SIZE} for Multi-Expert EfficientNet", 
                        width=224
                    )
                    
                    # Show coordinates info
                    if "patch_coords" in st.session_state:
                        coords = st.session_state["patch_coords"]
                        x1, y1, x2, y2 = coords
                        
                        # Check if whole image is selected
                        if hasattr(st.session_state, 'current_image'):
                            image_width, image_height = st.session_state.current_image.size
                            is_whole_image = (x1 == 0 and y1 == 0 and x2 == image_width and y2 == image_height)
                            
                            if is_whole_image:
                                st.caption(f"**Analysis Type:** Whole Image")
                                st.caption(f"**Original Size:** {image_width} × {image_height} pixels")
                            else:
                                st.caption(f"**Analysis Type:** Selected Region")
                                st.caption(f"**Region:** ({x1}, {y1}) to ({x2}, {y2})")
                                st.caption(f"**Original Size:** {x2-x1} × {y2-y1} pixels")
                
                with col_results2:
                    # Classify patch with comprehensive analysis
                    with st.spinner("🧠 Analyzing with Multi-Expert EfficientNet..."):
                        try:
                            transform = get_transforms()
                            patch_tensor = transform(st.session_state["current_patch"]).unsqueeze(0).to(device)
                            
                            # Get detailed analysis
                            analysis = model.get_expert_analysis(patch_tensor)
                            
                            # Get final prediction
                            final_probs = analysis['final_probs'][0]
                            predicted_class_idx = np.argmax(final_probs)
                            predicted_class = CLASS_NAMES[predicted_class_idx]
                            confidence = analysis['confidence'][0][0]
                            
                            # Generate XAI visualizations
                            gradcam = GradCAM(model)
                            gradcam_heatmap = gradcam.generate_cam(patch_tensor, predicted_class_idx)
                            attention_heatmap = create_attention_heatmap(model, patch_tensor)
                            
                            # Display result
                            if predicted_class == 'normal':
                                st.success(f"{CLASS_EMOJIS[predicted_class]} **{predicted_class.upper()}** - Confidence: {final_probs[predicted_class_idx]:.1%}")
                            elif predicted_class == 'benign':
                                st.warning(f"{CLASS_EMOJIS[predicted_class]} **{predicted_class.upper()}** - Confidence: {final_probs[predicted_class_idx]:.1%}")
                            else:  # malignant
                                st.error(f"{CLASS_EMOJIS[predicted_class]} **{predicted_class.upper()}** - Confidence: {final_probs[predicted_class_idx]:.1%}")
                            
                            # Show all class probabilities
                            st.subheader("📊 Class Probabilities")
                            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, final_probs)):
                                st.metric(
                                    label=f"{CLASS_EMOJIS[class_name]} {class_name.title()}",
                                    value=f"{prob:.1%}",
                                    delta=f"Rank: {np.argsort(final_probs)[::-1].tolist().index(i) + 1}"
                                )
                            
                            # Store comprehensive results for dashboard
                            st.session_state.analysis_results = analysis
                            st.session_state.predicted_class = predicted_class
                            st.session_state.model_confidence = confidence
                            st.session_state.gradcam_heatmap = gradcam_heatmap
                            st.session_state.attention_heatmap = attention_heatmap
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {e}")
                            st.exception(e)
            
            elif st.session_state.get("coordinates"):
                st.info("👆 Click 'Analyze Region' to start AI classification")
            else:
                st.info("� Draw a rectangle on the mammogram to select a region for analysis")
    
    with tab2:
        st.subheader("📊 Comprehensive Analysis Dashboard")
        
        if hasattr(st.session_state, 'analysis_results'):
            # Create tabs within the dashboard
            dashboard_tabs = st.tabs([
                "🧠 Expert Analysis", 
                "🔍 XAI Breakdown", 
                "📈 Technical Metrics",
                "💾 Export Results"
            ])
            
            with dashboard_tabs[0]:
                # Expert Network Analysis
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if show_expert_analysis:
                        st.subheader("🧠 Multi-Expert Network Analysis")
                        expert_fig = create_expert_analysis_plot(st.session_state.analysis_results)
                        st.plotly_chart(expert_fig, use_container_width=True)
                    
                    # Expert contributions table
                    st.subheader("📋 Expert Contributions Detail")
                    gate_weights = st.session_state.analysis_results['gate_weights'][0]
                    expert_probs = st.session_state.analysis_results['expert_probs']
                    
                    expert_data = []
                    for i in range(NUM_EXPERTS):
                        expert_data.append({
                            'Expert': f'Expert {i+1}',
                            'Type': ['Deep Narrow', 'Wide Shallow', 'Residual', 'Dense Attention'][i],
                            'Gate Weight': f"{gate_weights[i]:.1%}",
                            'Normal': f"{expert_probs[i][0][2]:.1%}",
                            'Benign': f"{expert_probs[i][0][0]:.1%}",
                            'Malignant': f"{expert_probs[i][0][1]:.1%}"
                        })
                    
                    df_experts = pd.DataFrame(expert_data)
                    st.dataframe(df_experts, use_container_width=True)
                
                with col2:
                    if show_confidence_gauge:
                        st.subheader("🎯 Model Confidence")
                        confidence_fig = create_confidence_gauge(st.session_state.model_confidence)
                        st.plotly_chart(confidence_fig, use_container_width=True)
                    
                    # Model statistics
                    st.subheader("📈 Model Statistics")
                    st.metric("Prediction", st.session_state.predicted_class.title())
                    st.metric("Overall Confidence", f"{st.session_state.model_confidence:.1%}")
                    st.metric("Expert Consensus", f"{np.std(gate_weights):.3f}")
            
            with dashboard_tabs[1]:
                # XAI Analysis
                if hasattr(st.session_state, 'gradcam_heatmap') and hasattr(st.session_state, 'attention_heatmap'):
                    st.subheader("🔍 Explainable AI (XAI) Analysis")
                    
                    # Comprehensive XAI visualization
                    xai_fig = create_comprehensive_xai_analysis(
                        st.session_state.analysis_results,
                        st.session_state.current_patch,
                        st.session_state.gradcam_heatmap,
                        st.session_state.attention_heatmap
                    )
                    st.plotly_chart(xai_fig, use_container_width=True)
                    
                    # Individual XAI components
                    col_xai1, col_xai2, col_xai3 = st.columns(3)
                    
                    with col_xai1:
                        st.subheader("🔥 GradCAM Analysis")
                        gradcam_overlay = overlay_heatmap_on_image(
                            st.session_state.current_patch, 
                            st.session_state.gradcam_heatmap, 
                            alpha=0.6
                        )
                        st.image(gradcam_overlay, caption="GradCAM: Areas important for prediction", width=224)
                        
                        # GradCAM metrics
                        gradcam_max = np.max(st.session_state.gradcam_heatmap)
                        gradcam_mean = np.mean(st.session_state.gradcam_heatmap)
                        st.metric("Peak Activation", f"{gradcam_max:.3f}")
                        st.metric("Average Activation", f"{gradcam_mean:.3f}")
                    
                    with col_xai2:
                        st.subheader("🎯 Feature Attention")
                        attention_overlay = overlay_heatmap_on_image(
                            st.session_state.current_patch, 
                            st.session_state.attention_heatmap, 
                            alpha=0.6,
                            colormap_name='plasma'
                        )
                        st.image(attention_overlay, caption="Feature Attention: Model focus areas", width=224)
                        
                        # Attention metrics
                        attention_std = np.std(st.session_state.attention_heatmap)
                        attention_coverage = np.sum(st.session_state.attention_heatmap > 0.5) / st.session_state.attention_heatmap.size
                        st.metric("Attention Spread", f"{attention_std:.3f}")
                        st.metric("Focus Coverage", f"{attention_coverage:.1%}")
                    
                    with col_xai3:
                        st.subheader("📊 XAI Summary")
                        
                        # Calculate interpretability metrics
                        gradcam_focus = np.max(st.session_state.gradcam_heatmap)
                        attention_dispersion = np.std(st.session_state.attention_heatmap)
                        expert_agreement = len(set(np.argmax([ep[0] for ep in st.session_state.analysis_results['expert_probs']], axis=1)))
                        
                        # Display metrics
                        st.metric("GradCAM Focus", f"{gradcam_focus:.3f}", 
                                help="Higher values indicate more focused attention")
                        st.metric("Attention Spread", f"{attention_dispersion:.3f}",
                                help="Higher values indicate more distributed attention")
                        st.metric("Expert Agreement", f"{expert_agreement}/{NUM_EXPERTS}",
                                help="Number of experts predicting the same class")
                        
                        # Interpretability score
                        interpretability_score = (gradcam_focus * 0.4 + 
                                                (1 - attention_dispersion) * 0.3 + 
                                                (expert_agreement / NUM_EXPERTS) * 0.3)
                        
                        st.metric("Interpretability Score", f"{interpretability_score:.3f}",
                                help="Combined measure of model explainability (0-1)")
                        
                        # Interpretation guidance
                        if interpretability_score > 0.7:
                            st.success("🟢 High interpretability - Model decision is well-explained")
                        elif interpretability_score > 0.5:
                            st.warning("🟡 Medium interpretability - Some uncertainty in explanation")
                        else:
                            st.error("🔴 Low interpretability - Model decision is less clear")
                
                else:
                    st.info("🔍 XAI analysis will appear here after analyzing an ROI.")
            
            with dashboard_tabs[2]:
                # Technical Metrics
                st.subheader("🔧 Technical Analysis")
                
                col_tech1, col_tech2 = st.columns(2)
                
                with col_tech1:
                    st.subheader("🧮 Computational Metrics")
                    
                    # Model architecture info
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    st.metric("Total Parameters", f"{total_params:,}")
                    st.metric("Trainable Parameters", f"{trainable_params:,}")
                    st.metric("Model Size", f"~{total_params * 4 / 1e6:.1f} MB")
                    st.metric("Device", str(device).upper())
                    
                    # Analysis timing
                    if hasattr(st.session_state, 'analysis_timestamp'):
                        analysis_time = time.time() - st.session_state.analysis_timestamp
                        st.metric("Analysis Time", f"{analysis_time:.2f}s")
                
                with col_tech2:
                    st.subheader("📊 Prediction Statistics")
                    
                    final_probs = st.session_state.analysis_results['final_probs'][0]
                    
                    # Statistical measures
                    entropy = -np.sum(final_probs * np.log(final_probs + 1e-10))
                    max_prob = np.max(final_probs)
                    prob_std = np.std(final_probs)
                    
                    st.metric("Prediction Entropy", f"{entropy:.3f}",
                            help="Lower values indicate more confident predictions")
                    st.metric("Max Probability", f"{max_prob:.1%}",
                            help="Highest class probability")
                    st.metric("Probability Spread", f"{prob_std:.3f}",
                            help="Standard deviation of class probabilities")
                    
                    # Certainty assessment
                    if max_prob > 0.8 and entropy < 0.5:
                        st.success("🎯 High Certainty Prediction")
                    elif max_prob > 0.6 and entropy < 1.0:
                        st.warning("⚖️ Moderate Certainty")
                    else:
                        st.error("❓ Low Certainty - Consider additional analysis")
                
                # Technical details expandable section
                with st.expander("🔧 Advanced Technical Details"):
                    st.code(f"""
Multi-Expert EfficientNet Architecture Details:
=============================
Backbone: EfficientNet-B1 (Pre-trained on ImageNet)
Expert Networks: {NUM_EXPERTS} specialized architectures
- Expert 1: Deep Narrow (256→128→classes)
- Expert 2: Wide Shallow (512→classes)
- Expert 3: Residual (384→384→classes with skip)
- Expert 4: Dense Attention (256→256→classes)

Gating Network: Intelligent routing (256→128→{NUM_EXPERTS})
Feature Enhancement: Linear projection with BatchNorm
Global Pooling: Adaptive Average Pooling 2D

Input Resolution: {IMG_SIZE}×{IMG_SIZE} RGB
Normalization: ImageNet standards
Device: {device}
Mixed Precision: Enabled for training
                    """)
            
            with dashboard_tabs[3]:
                # Export Results
                st.subheader("💾 Export Analysis Results")
                
                # Prepare export data
                export_data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction': {
                        'class': st.session_state.predicted_class,
                        'confidence': float(st.session_state.model_confidence),
                        'probabilities': {
                            CLASS_NAMES[i]: float(prob) 
                            for i, prob in enumerate(st.session_state.analysis_results['final_probs'][0])
                        }
                    },
                    'expert_analysis': {
                        'gate_weights': [float(w) for w in st.session_state.analysis_results['gate_weights'][0]],
                        'expert_predictions': [
                            {CLASS_NAMES[j]: float(prob) for j, prob in enumerate(expert_prob[0])}
                            for expert_prob in st.session_state.analysis_results['expert_probs']
                        ]
                    },
                    'roi_coordinates': st.session_state.patch_coords if hasattr(st.session_state, 'patch_coords') else None
                }
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    # JSON export
                    st.subheader("📄 JSON Report")
                    export_json = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=export_json,
                        file_name=f"moeffnet_analysis_{int(time.time())}.json",
                        mime="application/json"
                    )
                    
                    # Show preview
                    with st.expander("👀 Preview JSON"):
                        st.json(export_data)
                
                with col_export2:
                    # Summary report
                    st.subheader("📋 Summary Report")
                    
                    summary_text = f"""
Multi-Expert EfficientNet Breast Cancer Analysis Report
=====================================
Analysis Date: {export_data['timestamp']}

PREDICTION RESULTS:
- Classification: {export_data['prediction']['class'].upper()}
- Confidence: {export_data['prediction']['confidence']:.1%}
- Normal: {export_data['prediction']['probabilities']['normal']:.1%}
- Benign: {export_data['prediction']['probabilities']['benign']:.1%}
- Malignant: {export_data['prediction']['probabilities']['malignant']:.1%}

EXPERT ANALYSIS:
- Expert 1 Weight: {export_data['expert_analysis']['gate_weights'][0]:.1%}
- Expert 2 Weight: {export_data['expert_analysis']['gate_weights'][1]:.1%}
- Expert 3 Weight: {export_data['expert_analysis']['gate_weights'][2]:.1%}
- Expert 4 Weight: {export_data['expert_analysis']['gate_weights'][3]:.1%}

TECHNICAL INFO:
- Model: Multi-Expert EfficientNet-B1
- Input Size: 224×224 pixels
- Device: {device}
- Analysis Method: Deep Learning with XAI

DISCLAIMER:
This analysis is for research purposes only.
Not intended for clinical diagnosis.
Always consult healthcare professionals.
                    """
                    
                    st.download_button(
                        label="📄 Download Summary Report",
                        data=summary_text,
                        file_name=f"moeffnet_summary_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                    
                    st.text_area("📋 Report Preview", summary_text, height=400)
        
        else:
            st.info("📸 Upload and analyze an image to see comprehensive results here.")
            
            # Show what will be available
            st.markdown("""
            ### 📊 Available Analysis Features:
            
            **🧠 Expert Analysis:**
            - Multi-expert prediction breakdown
            - Gating weights visualization
            - Expert consensus analysis
            
            **🔍 XAI Breakdown:**
            - GradCAM attention maps
            - Feature attention visualization
            - Interpretability scoring
            
            **📈 Technical Metrics:**
            - Model architecture details
            - Computational statistics
            - Prediction confidence measures
            
            **💾 Export Options:**
            - JSON detailed report
            - Summary text report
            - Analysis timestamps and metadata
            """)
    
    with tab3:
        st.subheader("📖 About Multi-Expert EfficientNet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🏗️ Architecture Overview
            
            **Multi-Expert EfficientNet** combines multiple specialized expert networks with intelligent gating for superior breast cancer classification:
            
            1. **EfficientNet-B1 Backbone**: Pre-trained feature extractor
            2. **4 Expert Networks**: Each specialized for different pattern types
            3. **Gating Network**: Intelligently weights expert contributions
            4. **Confidence Estimation**: Provides reliability scores
            
            #### 🧠 Expert Specializations
            
            - **Expert 1 (Deep Narrow)**: Focused feature learning
            - **Expert 2 (Wide Shallow)**: Broad pattern recognition  
            - **Expert 3 (Residual)**: Skip connections for gradient flow
            - **Expert 4 (Dense Attention)**: Enhanced feature attention
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 Key Benefits
            
            - **Ensemble Learning**: Multiple perspectives on each patch
            - **Intelligent Routing**: Automatic expert selection per input
            - **Interpretability**: Visualize expert contributions
            - **Robustness**: Better generalization through diversity
            - **Confidence Aware**: Built-in uncertainty estimation
            
            #### 📊 Performance Characteristics
            
            - **High Accuracy**: Ensemble improves individual expert performance
            - **Low Variance**: Multiple experts reduce prediction uncertainty
            - **Explainable**: Expert analysis provides insight into decisions
            - **Scalable**: Easy to add/remove experts as needed
            """)
        
        st.markdown("""
        ---
        #### 🔬 Technical Implementation
        
        This application implements the complete Multi-Expert EfficientNet pipeline from the research notebook:
        
        1. **Professional Data Pipeline**: Custom Dataset and optimized DataLoader
        2. **Advanced Training**: Mixed precision, early stopping, comprehensive metrics
        3. **Expert Analysis**: Detailed visualization of expert contributions
        4. **Interactive Interface**: Real-time mammogram analysis with click selection
        5. **Production Ready**: Cached model loading, error handling, professional UI
        
        The model achieves state-of-the-art performance through its multi-expert architecture
        while maintaining interpretability through gating visualization.
        """)

if __name__ == "__main__":
    main()
