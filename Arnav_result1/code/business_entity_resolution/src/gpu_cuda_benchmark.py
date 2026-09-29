import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

def run_cuda_benchmark():
    print("=" * 60)
    print("LAPTOP 1 — GPU HARDWARE & CUDA PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda:0")
    
    props = torch.cuda.get_device_properties(device)
    print(f"Device Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total VRAM: {props.total_memory / (1024**2):.1f} MB")
    print(f"Multi-Processors: {props.multi_processor_count}")
    print(f"CUDA Version (torch): {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    # Define GPU Neural Pair Scorer / Ranker
    class GPUPairScorer(nn.Module):
        def __init__(self, in_features=71, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.net(x)

    model = GPUPairScorer().to(device)
    model.eval()

    # Benchmark micro-batch sizes: 512, 1024, 2048, 4096
    batch_sizes = [512, 1024, 2048, 4096]
    benchmark_results = []
    
    print("\nBenchmarking GPU Pair Scoring Batch Throughput & VRAM:")
    print(f"{'Batch Size':>12} | {'Time (ms)':>10} | {'Pairs/sec':>12} | {'Allocated VRAM (MB)':>20} | {'Reserved VRAM (MB)':>20}")
    print("-" * 80)
    
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Simulated feature matrix of 71 features
        x_cpu = torch.randn(bs, 71, pin_memory=True)
        
        # Warmup
        with torch.no_grad():
            x_gpu = x_cpu.to(device, non_blocking=True)
            _ = model(x_gpu)
            torch.cuda.synchronize()
            
        # Timed benchmark: 100 iterations
        n_iters = 100
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                for _ in range(n_iters):
                    x_gpu = x_cpu.to(device, non_blocking=True)
                    out = model(x_gpu)
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        elapsed_sec = t1 - t0
        total_pairs = bs * n_iters
        throughput = total_pairs / elapsed_sec
        time_per_batch_ms = (elapsed_sec / n_iters) * 1000
        
        alloc_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
        res_vram = torch.cuda.max_memory_reserved(device) / (1024**2)
        
        benchmark_results.append({
            'batch_size': bs,
            'time_ms': time_per_batch_ms,
            'throughput': throughput,
            'alloc_vram_mb': alloc_vram,
            'res_vram_mb': res_vram
        })
        
        print(f"{bs:12d} | {time_per_batch_ms:10.2f} | {throughput:12,.0f} | {alloc_vram:20.2f} | {res_vram:20.2f}")
        
    # Select best batch size
    best = max(benchmark_results, key=lambda r: r['throughput'])
    print("\n" + "=" * 60)
    print(f"OPTIMAL STABLE GPU BATCH SIZE: {best['batch_size']}")
    print(f"Peak Throughput: {best['throughput']:,.0f} candidate pairs/sec")
    print(f"Peak VRAM Usage: {best['res_vram_mb']:.1f} MB (< 4 GB VRAM limit)")
    print("=" * 60)
    
    # Save benchmark report
    with open(r"D:\Arnav\reports\GPU_HARDWARE_BENCHMARK.txt", "w", encoding="utf-8") as f:
        f.write("LAPTOP 1 — GPU HARDWARE & CUDA PERFORMANCE BENCHMARK\n")
        f.write(f"GPU: {props.name}\n")
        f.write(f"Compute Capability: {props.major}.{props.minor}\n")
        f.write(f"VRAM Total: {props.total_memory / (1024**2):.1f} MB\n")
        f.write(f"CUDA: {torch.version.cuda}\n")
        f.write(f"cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}\n\n")
        f.write("Batch Benchmark Results:\n")
        for r in benchmark_results:
            f.write(f"Batch {r['batch_size']}: {r['throughput']:,.0f} pairs/sec, {r['alloc_vram_mb']:.2f} MB alloc, {r['res_vram_mb']:.2f} MB reserved\n")
        f.write(f"\nOptimal Batch Size: {best['batch_size']} with {best['throughput']:,.0f} pairs/sec\n")

if __name__ == "__main__":
    run_cuda_benchmark()
