"""
Reproduce R example in Python: CV with predictions and plots.

Reproduces the R code:
  setwd("~/Projects/hapc")
  n <- 200
  p <- 1
  X <- matrix(runif(n*p, -1, 1), n, p)
  Y <- 2*sin(8*pi*(X[,1]^2))/X[,1] + 10 + rnorm(n, 0, 1)
  
  Xnew=seq(-1,1,length.out=100)
  
  rescv <- cv.hapc(X, Y,
                   npcs = n,
                   log_lambda_min =-5,
                   log_lambda_max = -3,
                   norm = "sv",
                   predict=Xnew,center=1,max_iter=200,tol=0.5,crit='grad',step_factor=1
  )
  
  plot(rescv$lambdas,rescv$mses)
  plot(X,Y,col='red')
  lines(Xnew,rescv$predictions,lwd=3)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add hapc to path
import hapc
from hapc.cv import cv_hapc

def main():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate data (matching R code)
    n = 200
    p = 1
    X = np.random.uniform(-1, 1, size=(n, p))
    Y = 2 * np.sin(8 * np.pi * (X[:, 0]**2)) / X[:, 0] + 10 + np.random.randn(n)
    
    print("="*70)
    print("Python Reproduction of R CV + Prediction Example")
    print("="*70)
    print(f"\nData shape: X {X.shape}, Y {Y.shape}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Y range: [{Y.min():.3f}, {Y.max():.3f}]")
    
    # Create prediction grid (matching R: seq(-1,1,length.out=100))
    Xnew = np.linspace(-1, 1, 100).reshape(-1, 1)
    print(f"Prediction grid shape: {Xnew.shape}")
    
    # Run CV with gradient descent (norm="sv")
    print("\n" + "="*70)
    print("Running CV (norm='sv', PC-GHAL Gradient Descent)")
    print("="*70)
    
    rescv = cv_hapc(
        X, Y,
        maxdeg=1,                          # default
        npc=n,                             # Request n, will be capped to n-1
        log_lambda_min=-6,                 # R: log_lambda_min = -5
        log_lambda_max=-3,                 # R: log_lambda_max = -3
        grid_length=10,                    # Generate 10 lambda values
        nfolds=5,                          # default
        norm="sv",                         # PC-GHAL gradient descent
        predict=Xnew,                      # Provide predictions
        center=True,
        max_iter=200,
        tol=0.5,
        verbose=True
    )
    
    print(f"\n✓ CV Complete!")
    print(f"Best lambda: {rescv.best_lambda:.6f}")
    best_mse = rescv.mses[np.argmin(rescv.mses)]
    print(f"Best MSE: {best_mse:.6f}")
    print(f"Number of alphas: {len(rescv.best_model_alpha)}")
    print(f"Predictions shape: {rescv.predictions.shape if rescv.predictions is not None else 'None'}")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MSE vs Lambdas
    ax = axes[0]
    ax.plot(rescv.lambdas, rescv.mses, 'b.-', linewidth=2, markersize=8)
    ax.axvline(rescv.best_lambda, color='r', linestyle='--', linewidth=2, label=f'Best λ={rescv.best_lambda:.6f}')
    ax.set_xlabel('Lambda (regularization)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Cross-Validation: MSE vs Lambda', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Plot 2: Data and predictions
    ax = axes[1]
    ax.scatter(X, Y, color='red', s=50, alpha=0.6, label='Training data')
    
    if rescv.predictions is not None:
        ax.plot(Xnew, rescv.predictions, 'b-', linewidth=3, label='Fitted curve (CV predictions)')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Data and Fitted Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = Path(__file__).parent / "cv_sv_example_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_file}")
    
    # Show statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Lambda grid: {rescv.lambdas}")
    print(f"MSE values: {rescv.mses}")
    best_mse = rescv.mses[np.argmin(rescv.mses)]
    print(f"Best lambda: {rescv.best_lambda:.6f}")
    print(f"Best MSE: {best_mse:.6f}")
    
    if rescv.predictions is not None:
        print(f"\nPredictions on Xnew:")
        print(f"  Min: {rescv.predictions.min():.3f}")
        print(f"  Max: {rescv.predictions.max():.3f}")
        print(f"  Mean: {rescv.predictions.mean():.3f}")
    
    plt.show()


if __name__ == "__main__":
    main()
