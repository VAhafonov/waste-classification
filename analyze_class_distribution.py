"""
Calculate class distribution and weights for weighted cross-entropy loss
"""

import argparse
import numpy as np
from collections import Counter

from utils.class_mapping import idx_to_class_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split_file', type=str, help='Path to train split file')
    args = parser.parse_args()
    
    # Count classes
    class_counts = Counter()
    with open(args.split_file, 'r') as f:
        for line in f:
            line = line.strip()
            class_idx = int(line[-1])  # Last character is class index
            class_counts[class_idx] += 1
    
    # Convert to array
    counts = np.array([class_counts[i] for i in range(9)])  # 9 classes
    total = counts.sum()
    
    # Calculate weights
    weights = total / (9 * counts)
    weights = weights / weights.mean()  # normalize
    
    # Print results
    print("Class Distribution:")
    for i, (count, weight) in enumerate(zip(counts, weights)):
        class_name = idx_to_class_name[i]
        print(f"{class_name:15} {count:5d} ({count/total*100:5.1f}%) weight: {weight:.3f}")
    
    print(f"\nTotal samples: {total}")
    print(f"Weights tensor: {weights.tolist()}")


if __name__ == "__main__":
    main()
