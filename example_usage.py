"""
Example usage of the Labor Market Model

This script demonstrates how to use the labor market model with
a binding wage floor and dynamic adjustment.
"""

import numpy as np
import matplotlib.pyplot as plt
from labor_market_model import LaborMarketParams, LaborMarketModel


def main():
    # Example parameters
    # These should be chosen such that w_bar > w* (binding condition)
    
    params = LaborMarketParams(
        a_S=5.0,      # labor supply intercept
        b_S=0.5,      # labor supply slope
        a_D0=20.0,    # labor demand intercept at t=0
        b_D=1.0,      # labor demand slope
        k=3.0,        # magnitude of inward demand shift over time
        w_bar=12.0,   # binding wage floor
        t=0.0         # initial time (short run)
    )
    
    # Create model
    model = LaborMarketModel(params)
    
    # Print summary
    print(model.summary())
    
    # Plot initial state (t=0)
    print("\nGenerating plot for t=0 (short run)...")
    model.plot_market(t=0.0, save_path='labor_market_t0.png')
    plt.close()
    
    # Plot intermediate state (t=0.5)
    print("\nGenerating plot for t=0.5 (intermediate)...")
    model.plot_market(t=0.5, save_path='labor_market_t05.png')
    plt.close()
    
    # Plot long run (t=1.0)
    print("\nGenerating plot for t=1.0 (long run)...")
    model.plot_market(t=1.0, save_path='labor_market_t1.png')
    plt.close()
    
    # Plot dynamics over time
    print("\nGenerating dynamics plot...")
    model.plot_dynamics(num_points=100, save_path='labor_market_dynamics.png')
    plt.close()
    
    # Demonstrate comparative statics
    print("\n" + "="*60)
    print("Comparative Statics Over Time")
    print("="*60)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        L_t = model.employment_at_wage_floor(t)
        U_t = model.unemployment(t)
        print(f"t={t:.2f}: Employment={L_t:.4f}, Unemployment={U_t:.4f}")


if __name__ == "__main__":
    main()
