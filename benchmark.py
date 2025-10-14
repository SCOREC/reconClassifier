import time
import json
import torch
import platform

import numpy as np
from pathlib import Path

# Make psutil optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Hardware info will be limited.")
    print("Install with: pip install psutil")


class TrainingBenchmark:
    """
    Tracks and reports performance metrics during training.
    
    Measures:
    - Hardware specifications
    - Epoch timing
    - Batch processing throughput
    - GPU memory usage
    - Samples processed per second
    """
    
    def __init__(self, device, enabled=True):
        """
        Initialize benchmark tracker.
        
        Parameters:
        device: torch.device - The device being used for training
        enabled: bool - Whether benchmarking is active
        """
        self.device = device
        self.enabled = enabled
        
        if not enabled:
            return
            
        self.epoch_times = []
        self.batch_times = []
        self.samples_processed = 0
        self.epoch_start = None
        self.training_start = time.time()
        
        #collect hardware info
        self.hardware_info = self._collect_hardware_info()
        
    def _collect_hardware_info(self):
        """Collect system and GPU hardware information."""
        info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
        }
        
        # Add psutil info if available
        if PSUTIL_AVAILABLE:
            info['cpu_count'] = psutil.cpu_count(logical=False)
            info['cpu_count_logical'] = psutil.cpu_count(logical=True)
            info['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
        else:
            info['cpu_count'] = 'N/A (psutil not installed)'
            info['cpu_count_logical'] = 'N/A (psutil not installed)'
            info['ram_gb'] = 'N/A (psutil not installed)'
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_count'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['gpu_memory_gb'] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            
            #multi-GPU info
            if torch.cuda.device_count() > 1:
                info['all_gpus'] = [
                    {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_gb': round(
                            torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                        )
                    }
                    for i in range(torch.cuda.device_count())
                ]
        else:
            info['gpu_name'] = 'CPU only'
            info['gpu_count'] = 0
            
        return info
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        if not self.enabled:
            return
        self.epoch_start = time.time()
        
    def end_epoch(self):
        """Mark the end of an epoch and record timing."""
        if not self.enabled:
            return
        if self.epoch_start is not None:
            self.epoch_times.append(time.time() - self.epoch_start)
        
    def record_batch(self, batch_size, batch_time):
        """
        Record metrics for a single batch.
        
        Parameters:
        batch_size: int - Number of samples in the batch
        batch_time: float - Time taken to process the batch (seconds)
        """
        if not self.enabled:
            return
        self.batch_times.append(batch_time)
        self.samples_processed += batch_size
        
    def get_throughput(self):
        """
        Calculate average throughput.
        
        Returns:
        float - Samples processed per second
        """
        if not self.enabled or len(self.batch_times) == 0:
            return 0
        total_time = sum(self.batch_times)
        return self.samples_processed / total_time if total_time > 0 else 0
        
    def get_gpu_memory_usage(self):
        """
        Get current GPU memory usage.
        
        Returns:
        float - Current GPU memory allocated in GB
        """
        if not self.enabled:
            return 0
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0
    
    def get_peak_gpu_memory(self):
        """
        Get peak GPU memory usage.
        
        Returns:
        float - Peak GPU memory allocated in GB
        """
        if not self.enabled:
            return 0
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0
    
    def print_hardware_info(self):
        """Print hardware configuration at start of training."""
        if not self.enabled:
            return
            
        print("\n" + "="*70)
        print("HARDWARE CONFIGURATION")
        print("="*70)
        print(f"Platform:        {self.hardware_info['platform']}")
        print(f"CPU:             {self.hardware_info['processor']}")
        
        if PSUTIL_AVAILABLE:
            print(f"CPU Cores:       {self.hardware_info['cpu_count']} physical, "
                  f"{self.hardware_info['cpu_count_logical']} logical")
            print(f"RAM:             {self.hardware_info['ram_gb']:.2f} GB")
        else:
            print(f"CPU Cores:       {self.hardware_info['cpu_count']}")
            print(f"RAM:             {self.hardware_info['ram_gb']}")
            
        print(f"Python Version:  {self.hardware_info['python_version']}")
        print(f"PyTorch Version: {self.hardware_info['torch_version']}")
        
        if self.hardware_info['gpu_count'] > 0:
            print(f"\nGPU Information:")
            print(f"  Primary GPU:   {self.hardware_info['gpu_name']}")
            print(f"  GPU Count:     {self.hardware_info['gpu_count']}")
            print(f"  VRAM:          {self.hardware_info['gpu_memory_gb']:.2f} GB")
            print(f"  CUDA Version:  {self.hardware_info['cuda_version']}")
            print(f"  cuDNN Version: {self.hardware_info['cudnn_version']}")
            
            if 'all_gpus' in self.hardware_info:
                print(f"\n  All GPUs:")
                for gpu in self.hardware_info['all_gpus']:
                    print(f"    [{gpu['id']}] {gpu['name']} ({gpu['memory_gb']:.2f} GB)")
        else:
            print("\nGPU:             Not available (CPU only)")
        
        print("="*70 + "\n")
    
    def print_summary(self, output_file=None):
        """
        Print comprehensive benchmark summary.
        
        Parameters:
        output_file: Path or str - Optional file path to save JSON benchmark data
        """
        if not self.enabled:
            return
            
        total_training_time = time.time() - self.training_start
        
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if len(self.epoch_times) > 0:
            print(f"\nTraining Performance:")
            print(f"  Total epochs completed:    {len(self.epoch_times)}")
            print(f"  Total training time:       {total_training_time:.2f}s "
                  f"({total_training_time/60:.2f} min)")
            print(f"  Average epoch time:        {np.mean(self.epoch_times):.2f}s")
            print(f"  Fastest epoch:             {np.min(self.epoch_times):.2f}s")
            print(f"  Slowest epoch:             {np.max(self.epoch_times):.2f}s")
            print(f"  Epoch time std dev:        {np.std(self.epoch_times):.2f}s")
            
        if len(self.batch_times) > 0:
            print(f"\nThroughput Metrics:")
            print(f"  Total samples processed:   {self.samples_processed:,}")
            print(f"  Total batches processed:   {len(self.batch_times):,}")
            print(f"  Average batch time:        {np.mean(self.batch_times)*1000:.2f}ms")
            print(f"  Throughput:                {self.get_throughput():.2f} samples/sec")
            print(f"  Time per sample:           {1000/self.get_throughput():.2f}ms")
            
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage:")
            print(f"  Current allocation:        {self.get_gpu_memory_usage():.2f} GB")
            print(f"  Peak allocation:           {self.get_peak_gpu_memory():.2f} GB")
            print(f"  Total GPU memory:          {self.hardware_info['gpu_memory_gb']:.2f} GB")
            print(f"  Peak utilization:          "
                  f"{100*self.get_peak_gpu_memory()/self.hardware_info['gpu_memory_gb']:.1f}%")
        
        print("="*70 + "\n")
        
        if output_file is not None:
            self.save_json(output_file)
    
    def save_json(self, output_file):
        """
        Save benchmark data to JSON file.
        
        Parameters:
        output_file: Path or str - File path to save benchmark data
        """
        if not self.enabled:
            return
            
        benchmark_data = {
            'hardware': self.hardware_info,
            'training': {
                'total_epochs': len(self.epoch_times),
                'total_training_time_sec': time.time() - self.training_start,
                'epoch_times_sec': self.epoch_times,
                'avg_epoch_time_sec': float(np.mean(self.epoch_times)) if self.epoch_times else 0,
                'min_epoch_time_sec': float(np.min(self.epoch_times)) if self.epoch_times else 0,
                'max_epoch_time_sec': float(np.max(self.epoch_times)) if self.epoch_times else 0,
            },
            'throughput': {
                'total_samples': self.samples_processed,
                'total_batches': len(self.batch_times),
                'avg_batch_time_sec': float(np.mean(self.batch_times)) if self.batch_times else 0,
                'samples_per_sec': self.get_throughput(),
            },
            'gpu_memory': {
                'peak_allocation_gb': self.get_peak_gpu_memory(),
                'final_allocation_gb': self.get_gpu_memory_usage(),
            } if torch.cuda.is_available() else None
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"Benchmark data saved to: {output_path}")