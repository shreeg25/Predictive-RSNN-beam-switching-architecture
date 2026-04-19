import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. DEFINE THE EXACT GRU ARCHITECTURE
# ==========================================
class GRUBeamTracker(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, output_dim=64):
        super(GRUBeamTracker, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        final_out = out[:, -1, :] 
        logits = self.fc(final_out)
        return logits

# ==========================================
# 2. BENCHMARK CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Benchmarking GRU Latency on: {device}")

# Load the model
model = GRUBeamTracker().to(device)
model.eval()

# Create a single dummy sequence matching your data pipeline
# Shape: [Batch Size=1, Sequence Length=10, Features=10]
dummy_input = torch.randn(1, 10, 10).to(device)

# ==========================================
# 3. WARM-UP PHASE
# ==========================================
# The first few GPU passes are always artificially slow due to memory allocation.
# We must warm up the GPU pipeline before starting the timer.
print("[*] Warming up the GPU (100 iterations)...")
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)

# ==========================================
# 4. MEASUREMENT PHASE
# ==========================================
ITERATIONS = 1000
latencies = []

print(f"[*] Executing {ITERATIONS} inference passes for statistical stability...")

# Use precise CUDA events if on GPU
if device.type == 'cuda':
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in range(ITERATIONS):
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            
            # Force PyTorch to wait for the GPU to physically finish the computation
            torch.cuda.synchronize() 
            
            # elapsed_time returns milliseconds
            latencies.append(start_event.elapsed_time(end_event))
else:
    # CPU fallback timing
    import time
    with torch.no_grad():
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # Convert to milliseconds

# ==========================================
# 5. RESULTS
# ==========================================
avg_latency = np.mean(latencies)
std_latency = np.std(latencies)

print("\n======================================")
print(f"[+] GRU Single-Inference Latency: {avg_latency:.4f} ms")
print(f"[+] Standard Deviation:         ± {std_latency:.4f} ms")
print("======================================")