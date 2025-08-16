# Docker Instructions

## Building the Docker Image

```bash
# Build the Docker image
docker build -t waste-classification .
```

## Running the Docker Container

### Basic Run with Volume Mounting

```bash
# Run with current directory mounted as workspace
docker run -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/saved_models:/workspace/saved_models \
  waste-classification
```

### Run with TensorBoard Port Exposed

```bash
# Run with TensorBoard port exposed for monitoring
docker run -it --ipc=host --shm-size=10gb \
  -p 6006:6006 \
  -v $(pwd):/workspace \
  -v /Users/vahaf/Projects/data/waste/RealWaste:/data \
  waste-classification
```


### Run Training Script

```bash
# Run training directly
docker run -it --ipc=host --shm-size=10gb \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/saved_models:/workspace/saved_models \
  waste-classification \
  python train.py --config config/train_config.yaml
```

### Start TensorBoard (inside container)

```bash
# Inside the container, start TensorBoard to monitor training
tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006
```

## Volume Explanations

- `/workspace`: Main working directory (maps to your project root)
- `/workspace/data`: Dataset storage
- `/workspace/logs`: TensorBoard logs and training logs
- `/workspace/saved_models`: Saved model checkpoints
