import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Mock os.sched_getaffinity for macOS compatibility
if not hasattr(os, 'sched_getaffinity'):
    os.sched_getaffinity = lambda x: [0]

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from utils import compute_alpha_exact

def plot_figure_9():
    print("Generating Figure 9 curves...")
    t_vals = np.linspace(0.0, 1.0, 500)
    # Clip slightly to avoid divide-by-zero inside standard deviation bounds in Gauss-Hermite
    t_vals_clipped = np.clip(t_vals, 1e-6, 1.0 - 1e-6)
    
    vocab_sizes = [10, 100, 1000, 30000, 50257]
    
    plt.figure(figsize=(12, 5))
    
    # Left subplot: Decoding error rate P_e(t) vs t
    plt.subplot(1, 2, 1)
    for K in vocab_sizes:
        alpha = compute_alpha_exact(t_vals_clipped, K=K)
        # Ensure exact boundaries
        alpha[0] = 0.0
        alpha[-1] = 1.0
        pe = 1.0 - alpha
        plt.plot(t_vals, pe, label=f"|V| = {K}")
        
    plt.xlabel("Flow Time t")
    plt.ylabel("Decoding Error Rate Pe(t)")
    plt.title("Decoding Error Rate vs. Flow Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right subplot: Reparameterization tau(t) vs t
    plt.subplot(1, 2, 2)
    for K in vocab_sizes:
        alpha = compute_alpha_exact(t_vals_clipped, K=K)
        alpha[0] = 0.0
        alpha[-1] = 1.0
        pe = 1.0 - alpha
        tau = 1.0 - (K / (K - 1.0)) * pe
        tau = np.clip(tau, 0.0, 1.0)
        plt.plot(t_vals, tau, label=f"|V| = {K}")
        
    plt.xlabel("Flow Time t")
    plt.ylabel("Reparameterized Time tau(t)")
    plt.title("Reparameterized Time vs. Flow Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure plot directory exists
    os.makedirs("plot", exist_ok=True)
    out_path = "plot/linearization_reparameterization.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved successfully to {out_path}!")

if __name__ == "__main__":
    plot_figure_9()
