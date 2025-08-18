"""
Gradio Demo for Waste Classification Model
"""

import os
# Force English locale for Gradio interface
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from models import create_model
from utils.class_mapping import idx_to_class_name


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_model_checkpoint(config, checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    model = create_model(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def get_inference_transforms():
    """Get inference transforms (no augmentation)"""
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


class WasteClassifier:
    def __init__(self, config_path, checkpoint_path, device='auto'):
        """Initialize the waste classifier"""
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load config and model
        self.config = load_config(config_path)
        self.model = load_model_checkpoint(self.config, checkpoint_path, self.device)
        self.transform = get_inference_transforms()
        
        # Get class names
        self.class_names = [idx_to_class_name[i] for i in range(self.config['model']['num_classes'])]
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.class_names}")
    
    def predict(self, image):
        """Predict waste class for an image"""
        try:
            # Preprocess image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
            # Get top predictions
            probs = probabilities.cpu().numpy()[0]
            predictions = {}
            
            for i, class_name in enumerate(self.class_names):
                predictions[class_name] = float(probs[i])
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {class_name: 0.0 for class_name in self.class_names}


def create_demo(config_path, checkpoint_path, device='auto', logo_path=None, enable_auth=False):
    """Create Gradio demo interface"""
    
    # Initialize classifier
    classifier = WasteClassifier(config_path, checkpoint_path, device)
    
    def classify_waste(image):
        """Classify waste image and return predictions"""
        predictions = classifier.predict(image)
        return predictions
    
    # Create interface using Blocks for better HTML control
    # Force English interface regardless of browser locale
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="🗂️ Waste Classification Demo",
        analytics_enabled=False  # Also disable analytics for privacy
    ) as demo:
        
        # Add logo if provided
        if logo_path and os.path.exists(logo_path):
            # Convert logo to base64 for embedding
            import base64
            with open(logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
            
            # Get file extension for proper MIME type
            file_ext = os.path.splitext(logo_path)[1].lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.svg': 'image/svg+xml',
                '.gif': 'image/gif'
            }.get(file_ext, 'image/png')
            
            # Add logo as HTML component
            gr.HTML(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:{mime_type};base64,{logo_data}" 
                     alt="Company Logo" 
                     style="max-height: 80px; max-width: 300px; height: auto; width: auto;">
            </div>
            """)
        
        # Title
        gr.Markdown("# 🗂️ Waste Classification Demo")
        
        # Description
        gr.Markdown("""
        Upload an image of waste to classify it into one of 9 categories:
        
        **cardboard**, **food organics**, **glass**, **metal**, **misc**, **paper**, **plastic**, **textile**, **vegetation**
        
        The model will show confidence scores for each category.
        """)
        
        # Main interface
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Waste Image")
                classify_btn = gr.Button("Classify Waste", variant="primary")
            
            with gr.Column():
                output_label = gr.Label(num_top_classes=9, label="Classification Results")
        
        # Event handler
        classify_btn.click(
            fn=classify_waste,
            inputs=input_image,
            outputs=output_label
        )
        
        # Also allow classification on image upload
        input_image.upload(
            fn=classify_waste,
            inputs=input_image,
            outputs=output_label
        )
        
        # About section
        gr.Markdown("""
        ### About this Demo
        This demo uses a deep learning model trained on waste classification data. 
        The model can classify waste into 9 different categories to help with sorting and recycling.
        
        **Categories:**
        - 📦 **Cardboard**: Cardboard boxes, packaging
        - 🍎 **Food Organics**: Food waste, organic matter
        - 🪟 **Glass**: Glass bottles, containers
        - 🔩 **Metal**: Metal cans, containers
        - 🗑️ **Misc**: Mixed waste items
        - 📄 **Paper**: Paper documents, newspapers
        - 🥤 **Plastic**: Plastic bottles, containers
        - 👕 **Textile**: Clothing, fabric items
        - 🌱 **Vegetation**: Plant matter, leaves
        
        ---
        *Powered by CloEE*
        """)
    
    return demo


def main():
    """Main function to run the demo"""
    import argparse
    
    # Hardcoded authentication credentials - CHANGE THESE!
    AUTH_USERNAME = "cloee"
    AUTH_PASSWORD = "cloee2025"
    
    parser = argparse.ArgumentParser(description='Waste Classification Gradio Demo')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--logo', type=str, default="icon.png",
                       help='Path to company logo image (PNG, JPG, SVG)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cpu, cuda, or auto')
    parser.add_argument('--no-auth', action='store_true',
                       help='Disable authentication (public access)')
    parser.add_argument('--share', action='store_true',
                       help='Create a public shareable link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the demo on')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if args.logo and not os.path.exists(args.logo):
        raise FileNotFoundError(f"Logo file not found: {args.logo}")
    
    # Setup authentication
    enable_auth = not args.no_auth
    auth_credentials = [(AUTH_USERNAME, AUTH_PASSWORD)] if enable_auth else None
    
    print(f"Starting Gradio demo...")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Logo: {args.logo if args.logo else 'None'}")
    print(f"Device: {args.device}")
    
    if enable_auth:
        print(f"Authentication: Enabled (Username: {AUTH_USERNAME})")
        print("⚠️  WARNING: Change the hardcoded credentials in the script for production use!")
    else:
        print("Authentication: Disabled (Public access)")
    
    # Create and launch demo
    demo = create_demo(args.config, args.checkpoint, args.device, args.logo, enable_auth)
    
    demo.launch(
        auth=auth_credentials,
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"  # Allow external connections
    )


if __name__ == "__main__":
    main()
