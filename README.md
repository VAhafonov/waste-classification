# Waste Classification with Deep Learning

A simple deep learning project for waste classification using PyTorch and ResNet34. This project supports both CPU and GPU training with Docker containerization.

## Features

- **ResNet34 Model**: Baseline model with options for pretrained or from-scratch training
- **9 Waste Classes**: cardboard, glass, metal, paper, plastic, trash, battery, biological, clothes
- **Docker Support**: CPU-based PyTorch in Ubuntu 22.04 container
- **TensorBoard Logging**: Real-time training monitoring
- **YAML Configuration**: Easy configuration management
- **Dummy Dataset**: Built-in dummy data generation for testing
- **Flexible Training**: Support for different optimizers, schedulers, and augmentations

## Project Structure

```
waste-classification/
├── Dockerfile                 # Docker container definition
├── docker-instructions.md     # Docker usage instructions
├── requirements.txt          # Python dependencies
├── config/
│   └── train_config.yaml    # Training configuration
├── dataset.py               # Dataset class with dummy functionality
├── models/
│   ├── __init__.py          # Models package
│   └── resnet34.py          # ResNet34 model implementation
├── train.py                # Main training script
├── data/                   # Data directory (mounted volume)
├── logs/                   # TensorBoard logs (mounted volume)
└── saved_models/           # Saved models (mounted volume)
```

## Quick Start

### 1. Build Docker Image

```bash
docker build -t waste-classification .
```

### 2. Run Training with Dummy Data

```bash
# Run training with dummy data
docker run --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/saved_models:/workspace/saved_models \
  waste-classification \
  python train.py --config config/train_config.yaml --data-mode dummy
```

### 3. Monitor Training with TensorBoard

```bash
# In another terminal, start TensorBoard
docker run --rm -p 6006:6006 \
  -v $(pwd)/logs:/workspace/logs \
  waste-classification \
  tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006
```

Then open http://localhost:6006 in your browser.

## Configuration

Edit `config/train_config.yaml` to customize training parameters:

```yaml
# Model Configuration
model:
  name: "resnet34"
  num_classes: 9
  pretrained: true  # Set to false for training from scratch

# Training Configuration
training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"  # or "sgd"

# Dataset Configuration
dataset:
  batch_size: 32
  image_size: 224
```

## Usage Examples

### Training with Different Configurations

```bash
# Train from scratch (no pretrained weights)
# First, edit config/train_config.yaml and set pretrained: false
docker run --rm \
  -v $(pwd):/workspace \
  waste-classification \
  python train.py --config config/train_config.yaml

# Train with real data (place images in data/class_name/ structure)
docker run --rm \
  -v $(pwd):/workspace \
  -v /path/to/your/data:/workspace/data \
  waste-classification \
  python train.py --data-mode real

# Force CPU training
docker run --rm \
  -v $(pwd):/workspace \
  waste-classification \
  python train.py --device cpu
```

### Testing Components

```bash
# Test dataset functionality
docker run --rm -v $(pwd):/workspace waste-classification python dataset.py

# Test model functionality  
docker run --rm -v $(pwd):/workspace waste-classification python models/resnet34.py
```

### Interactive Development

```bash
# Start interactive container
docker run -it --rm \
  -p 6006:6006 \
  -v $(pwd):/workspace \
  waste-classification \
  /bin/bash

# Inside container:
python train.py --config config/train_config.yaml
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

## Data Structure

For real data training, organize your dataset as follows:

```
data/
├── cardboard/
│   ├── image1.jpg
│   └── image2.jpg
├── glass/
│   ├── image1.jpg
│   └── image2.jpg
├── metal/
├── paper/
├── plastic/
├── trash/
├── battery/
├── biological/
└── clothes/
```

## Model Features

- **Pretrained Option**: Uses ImageNet pretrained weights for faster convergence
- **From Scratch**: Train without pretrained weights for full control
- **Flexible Head**: Easily configurable number of output classes
- **Feature Extraction**: Access intermediate features for analysis
- **Checkpoint Support**: Save and load model states

## Training Features

- **Multiple Optimizers**: Adam and SGD support
- **Learning Rate Scheduling**: Step and Cosine annealing schedulers
- **Data Augmentation**: Horizontal flip, rotation, color jitter
- **Automatic Checkpointing**: Save best and periodic checkpoints
- **TensorBoard Integration**: Real-time loss and accuracy monitoring

## Development

### Without Docker (Local Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --config config/train_config.yaml --data-mode dummy

# Start TensorBoard
tensorboard --logdir=logs
```

### Customizing the Model

The ResNet34 model can be easily modified in `models/resnet34.py`:

```python
# Create custom model
from models import ResNet34

model = ResNet34(num_classes=9, pretrained=True)

# Get model information
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
```

## Output Files

- **logs/**: TensorBoard logs with training metrics
- **saved_models/**: Saved model checkpoints
  - `best_model.pth`: Best validation accuracy model
  - `final_model.pth`: Final epoch model
  - `checkpoint_epoch_*.pth`: Periodic checkpoints

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in config
2. **No Data Found**: Check data directory structure
3. **Permission Errors**: Ensure proper volume mounting permissions

### Docker Issues

```bash
# Check if image built correctly
docker images | grep waste-classification

# Check container logs
docker logs <container_id>

# Clean up containers
docker system prune
```

## Requirements

- Docker
- 4GB+ RAM (for training)
- Python 3.8+ (for local development)
- PyTorch 2.0+ (CPU version included in Docker)

## License

This project is for educational purposes. Feel free to modify and extend for your needs.