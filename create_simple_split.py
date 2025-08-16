import os
import random
import argparse
from pathlib import Path

# Import existing class mapping
try:
    from utils.class_mapping import class_name_to_idx
except ImportError:
    print("Warning: Could not import class mapping from utils/class_mapping.py")
    print("Using directory structure to create mapping...")
    class_name_to_idx = None


def get_class_mapping(input_dir):
    """Get class mapping from utils/class_mapping.py or create from directory structure"""
    if class_name_to_idx is not None:
        return class_name_to_idx
    else:
        # Fallback: create mapping from directory structure
        input_path = Path(input_dir)
        class_dirs = sorted([d.name for d in input_path.iterdir() if d.is_dir()])
        return {class_name: idx for idx, class_name in enumerate(class_dirs)}


def create_train_val_split(input_dir, split_ratio=0.8, version="v1", seed=42):
    """Create train/val split and save as text files"""
    
    # Set random seed for reproducibility
    random.seed(seed)
    print(f"Using random seed: {seed}")
    
    input_path = Path(input_dir)
    
    # Get class mapping
    class_to_idx = get_class_mapping(input_dir)
    print(f"Found classes: {list(class_to_idx.keys())}")
    
    train_lines = []
    val_lines = []
    
    # Process each class directory
    for class_name, class_idx in class_to_idx.items():
        class_dir = input_path / class_name
        
        if not class_dir.is_dir():
            continue
            
        print(f"Processing class: {class_name} (index: {class_idx})")

        # Get all image files
        image_files = [
            f for f in class_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ]
        
        # Shuffle and split
        random.shuffle(image_files)
        split_point = int(len(image_files) * split_ratio)
        
        train_files = image_files[:split_point]
        val_files = image_files[split_point:]
        
        # Create relative paths and add to lists
        for f in train_files:
            relative_path = f"{class_name.capitalize()}/{f.name}"
            train_lines.append(f"{relative_path} {class_idx}")
        
        for f in val_files:
            relative_path = f"{class_name.capitalize()}/{f.name}"
            val_lines.append(f"{relative_path} {class_idx}")
        
        print(f"  {len(train_files)} train, {len(val_files)} val")
    
    # Shuffle the final lists to mix classes
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    
    # Write to files
    train_file = input_path / f"train_{version}.txt"
    val_file = input_path / f"val_{version}.txt"
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_lines) + '\n')
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_lines) + '\n')
    
    print(f"\nSplit complete!")
    print(f"Train: {len(train_lines)} samples -> {train_file}")
    print(f"Val: {len(val_lines)} samples -> {val_file}")
    
    # Show class mapping being used
    print(f"\nClass mapping used:")
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {class_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple Train/Val Split Script - Creates train.txt and val.txt files with relative paths and class indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_simple_split.py ./data
  python create_simple_split.py ./data --split_ratio 0.75
  python create_simple_split.py ./data --version v2
  python create_simple_split.py ./data --split_ratio 0.8 --version v2 --seed 123
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Input directory with class subdirectories"
    )
    
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version suffix for output files (default: v1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        parser.error(f"Input directory {args.input_dir} does not exist")
    
    if not 0 < args.split_ratio < 1:
        parser.error(f"Split ratio must be between 0 and 1, got {args.split_ratio}")
    
    if args.seed < 0:
        parser.error(f"Seed must be non-negative, got {args.seed}")
    
    # Run the split
    create_train_val_split(args.input_dir, args.split_ratio, args.version, args.seed)


if __name__ == "__main__":
    main()