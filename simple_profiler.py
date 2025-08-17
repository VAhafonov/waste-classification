#!/usr/bin/env python3
"""
Simple Model Profiler for Waste Classification
Profiles one model at a time with 256x256 random images
"""

import argparse
import time
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import create_model


class SimpleProfiler:
    """Simple model profiler for single model analysis"""
    
    def __init__(self, device='auto', warmup_runs=10, profile_runs=50, half_precision=False):
        """Initialize profiler"""
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.warmup_runs = warmup_runs
        self.profile_runs = profile_runs
        self.half_precision = half_precision
        
        # Validate half precision settings
        if self.half_precision and self.device.type == 'cpu':
            print("⚠️  Warning: Half precision not recommended on CPU, switching to full precision")
            self.half_precision = False
        
        print(f"🚀 Simple Model Profiler")
        print(f"Device: {self.device}")
        print(f"Precision: {'Half (FP16)' if self.half_precision else 'Full (FP32)'}")
        print(f"Image size: 256x256")
        print("-" * 40)
    
    def count_parameters(self, model):
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_model_size_mb(self, model):
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def profile_inference(self, model, batch_size):
        """Profile inference time for given batch size"""
        model.eval()
        
        # Create random 256x256 RGB image
        dummy_input = torch.randn(batch_size, 3, 256, 256).to(self.device)
        
        # Convert to half precision if requested
        if self.half_precision:
            dummy_input = dummy_input.half()
        
        # Warmup
        print(f"  Warmup ({self.warmup_runs} runs)...", end="")
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)
        print(" ✓")
        
        # Timing runs
        print(f"  Profiling ({self.profile_runs} runs)...", end="")
        times = []
        
        with torch.no_grad():
            for _ in range(self.profile_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
        
        print(" ✓")
        
        # Calculate statistics
        times = np.array(times)
        return {
            'avg_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'throughput_fps': batch_size / np.mean(times)
        }
    
    def get_memory_usage(self, model, batch_size):
        """Get memory usage for given batch size"""
        dummy_input = torch.randn(batch_size, 3, 256, 256).to(self.device)
        
        # Convert to half precision if requested
        if self.half_precision:
            dummy_input = dummy_input.half()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
            
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            return memory_mb
        else:
            # For CPU, return approximate memory based on tensor sizes
            model_size = self.get_model_size_mb(model)
            bytes_per_element = 2 if self.half_precision else 4  # 2 bytes for FP16, 4 for FP32
            input_size = batch_size * 3 * 256 * 256 * bytes_per_element / 1024 / 1024
            return model_size + input_size * 2  # Rough estimate
    
    def profile_model(self, model_name, batch_sizes=[1, 4, 8, 16]):
        """Profile a single model"""
        print(f"\n🔍 Profiling: {model_name.upper()}")
        print("=" * 50)
        
        # Create model
        model_config = {
            'name': model_name,
            'num_classes': 9,
            'pretrained': True
        }
        
        try:
            # Load model
            print("Loading model...", end="")
            start = time.time()
            model = create_model(model_config)
            model = model.to(self.device)
            
            # Convert to half precision if requested
            if self.half_precision:
                try:
                    model = model.half()
                    print(" [FP16]", end="")
                except Exception as e:
                    print(f" ⚠️  Half precision conversion failed: {e}")
                    self.half_precision = False
                    print(" Falling back to FP32")
            
            loading_time = time.time() - start
            print(f" ✓ ({loading_time:.2f}s)")
            
            # Model info
            total_params, trainable_params = self.count_parameters(model)
            model_size = self.get_model_size_mb(model)
            
            print(f"Model Size: {model_size:.1f} MB")
            print(f"Parameters: {trainable_params:,} trainable, {total_params:,} total")
            print("-" * 50)
            
            # Profile each batch size
            results = {}
            for batch_size in batch_sizes:
                print(f"\nBatch Size: {batch_size}")
                
                try:
                    # Time profiling
                    timing_result = self.profile_inference(model, batch_size)
                    
                    # Memory profiling
                    memory_mb = self.get_memory_usage(model, batch_size)
                    
                    results[batch_size] = {
                        **timing_result,
                        'memory_mb': memory_mb
                    }
                    
                    print(f"  Time: {timing_result['avg_time_ms']:.1f}±{timing_result['std_time_ms']:.1f}ms")
                    print(f"  Throughput: {timing_result['throughput_fps']:.1f} FPS")
                    print(f"  Memory: {memory_mb:.1f} MB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ❌ Out of memory")
                        break
                    elif "half" in str(e).lower() or "float16" in str(e).lower():
                        print(f"  ❌ Half precision not supported for this operation")
                        if self.half_precision:
                            print("     Try running without --half_precision flag")
                        break
                    else:
                        raise e
                
                # Clear memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Summary
            self.print_summary(model_name, results)
            self.create_simple_plot(model_name, results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error profiling {model_name}: {e}")
            return None
        
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def print_summary(self, model_name, results):
        """Print summary of results"""
        if not results:
            return
        
        print(f"\n📊 SUMMARY - {model_name.upper()}")
        print("-" * 50)
        print(f"{'Batch':<8} {'Time (ms)':<12} {'FPS':<10} {'Memory (MB)':<12}")
        print("-" * 50)
        
        for batch_size, result in results.items():
            print(f"{batch_size:<8} {result['avg_time_ms']:<12.1f} {result['throughput_fps']:<10.1f} {result['memory_mb']:<12.1f}")
        
        # Best performance
        best_batch = min(results.keys(), key=lambda x: results[x]['avg_time_ms'])
        best_result = results[best_batch]
        
        print(f"\n🏆 Best Performance:")
        print(f"   Batch Size: {best_batch}")
        print(f"   Inference Time: {best_result['avg_time_ms']:.1f}ms")
        print(f"   Throughput: {best_result['throughput_fps']:.1f} FPS")
    
    def create_simple_plot(self, model_name, results):
        """Create simple visualization"""
        if len(results) < 2:
            return
        
        batch_sizes = list(results.keys())
        throughputs = [results[bs]['throughput_fps'] for bs in batch_sizes]
        memories = [results[bs]['memory_mb'] for bs in batch_sizes]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Throughput plot
        ax1.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (FPS)')
        ax1.set_title(f'{model_name} - Throughput')
        ax1.grid(True, alpha=0.3)
        
        # Memory plot
        ax2.plot(batch_sizes, memories, 's-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title(f'{model_name} - Memory Usage')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{model_name}_profile.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📈 Plot saved: {filename}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Simple model profiler')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet34', 'efficientnet_b0', 'mobilenet_v3_large'],
                       help='Model to profile')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 4, 8, 16],
                       help='Batch sizes to test (default: 1 4 8 16)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use (default: auto)')
    parser.add_argument('--warmup_runs', type=int, default=10,
                       help='Warmup runs (default: 10)')
    parser.add_argument('--profile_runs', type=int, default=50,
                       help='Profile runs (default: 50)')
    parser.add_argument('--half_precision', action='store_true',
                       help='Use half precision (FP16) for faster inference (GPU recommended)')
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = SimpleProfiler(
        device=args.device,
        warmup_runs=args.warmup_runs,
        profile_runs=args.profile_runs,
        half_precision=args.half_precision
    )
    
    # Profile the model
    results = profiler.profile_model(args.model, args.batch_sizes)
    
    if results:
        print(f"\n✅ Profiling completed for {args.model}!")
    else:
        print(f"\n❌ Profiling failed for {args.model}")


if __name__ == "__main__":
    main()
