"""
Perfectly Competitive Labor Market with a Binding Wage Floor and Dynamic Adjustment

This module implements a labor market model where:
- Labor markets are perfectly competitive
- A binding wage floor is imposed above the equilibrium wage
- Over time, firms adjust through capital substitution, automation, or exit,
  modeled as an inward shift of labor demand
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class LaborMarketParams:
    """Parameters for the labor market model"""
    a_S: float  # labor supply intercept
    b_S: float  # labor supply slope (must be > 0)
    a_D0: float  # labor demand intercept at t=0
    b_D: float  # labor demand slope (must be > 0)
    k: float  # magnitude of inward demand shift over time (>= 0)
    w_bar: float  # binding wage floor
    t: float = 0.0  # time index [0, 1] (0=short run, 1=long run)
    
    def __post_init__(self):
        """Validate parameters"""
        if self.b_S <= 0:
            raise ValueError("b_S must be > 0")
        if self.b_D <= 0:
            raise ValueError("b_D must be > 0")
        if self.k < 0:
            raise ValueError("k must be >= 0")
        if self.t < 0 or self.t > 1:
            raise ValueError("t must be in [0, 1]")


class LaborMarketModel:
    """
    Labor market model with binding wage floor and dynamic adjustment
    """
    
    def __init__(self, params: LaborMarketParams):
        self.params = params
        self._validate_binding_condition()
    
    def _validate_binding_condition(self):
        """Check that wage floor is binding (w_bar > w*)"""
        w_star = self.equilibrium_wage()
        if self.params.w_bar <= w_star:
            raise ValueError(
                f"Wage floor w_bar={self.params.w_bar} must be > "
                f"equilibrium wage w*={w_star:.4f} to be binding"
            )
    
    def labor_supply(self, L: float) -> float:
        """
        Labor supply function: w_S(L) = a_S + b_S * L
        
        Args:
            L: Labor quantity
            
        Returns:
            Wage at which L units of labor are supplied
        """
        return self.params.a_S + self.params.b_S * L
    
    def labor_demand_intercept(self, t: Optional[float] = None) -> float:
        """
        Labor demand intercept at time t: a_D(t) = a_D0 - k * t
        
        Args:
            t: Time index (defaults to self.params.t)
            
        Returns:
            Demand intercept at time t
        """
        if t is None:
            t = self.params.t
        return self.params.a_D0 - self.params.k * t
    
    def labor_demand(self, L: float, t: Optional[float] = None) -> float:
        """
        Labor demand function: w_D(L, t) = a_D(t) - b_D * L
        
        Args:
            L: Labor quantity
            t: Time index (defaults to self.params.t)
            
        Returns:
            Wage at which L units of labor are demanded
        """
        a_D = self.labor_demand_intercept(t)
        return a_D - self.params.b_D * L
    
    def equilibrium_labor(self) -> float:
        """
        Initial competitive equilibrium labor (pre-wage floor)
        L* = (a_D0 - a_S) / (b_S + b_D)
        
        Returns:
            Equilibrium labor quantity
        """
        return (self.params.a_D0 - self.params.a_S) / (
            self.params.b_S + self.params.b_D
        )
    
    def equilibrium_wage(self) -> float:
        """
        Initial competitive equilibrium wage (pre-wage floor)
        w* = a_S + b_S * L*
        
        Returns:
            Equilibrium wage
        """
        L_star = self.equilibrium_labor()
        return self.params.a_S + self.params.b_S * L_star
    
    def employment_at_wage_floor(self, t: Optional[float] = None) -> float:
        """
        Employment under the wage floor at time t
        L(t) = max{0, (a_D(t) - w_bar) / b_D}
        
        Args:
            t: Time index (defaults to self.params.t)
            
        Returns:
            Employment level at the wage floor
        """
        if t is None:
            t = self.params.t
        a_D = self.labor_demand_intercept(t)
        L_t = (a_D - self.params.w_bar) / self.params.b_D
        return max(0.0, L_t)
    
    def labor_supplied_at_wage_floor(self) -> float:
        """
        Labor supplied at the wage floor
        L_S(w_bar) = (w_bar - a_S) / b_S
        
        Returns:
            Labor quantity supplied at the wage floor
        """
        return (self.params.w_bar - self.params.a_S) / self.params.b_S
    
    def unemployment(self, t: Optional[float] = None) -> float:
        """
        Unemployment at time t
        U(t) = max{0, L_S(w_bar) - L(t)}
        
        Args:
            t: Time index (defaults to self.params.t)
            
        Returns:
            Unemployment level
        """
        L_S = self.labor_supplied_at_wage_floor()
        L_t = self.employment_at_wage_floor(t)
        return max(0.0, L_S - L_t)
    
    def employment_derivative(self) -> float:
        """
        Rate of change of employment over time
        dL(t)/dt = -k / b_D
        
        Returns:
            Derivative of employment with respect to time
        """
        return -self.params.k / self.params.b_D
    
    def plot_market(self, t: Optional[float] = None, 
                    L_max: Optional[float] = None,
                    save_path: Optional[str] = None):
        """
        Plot the labor market with supply, demand, wage floor, and unemployment
        
        Args:
            t: Time index (defaults to self.params.t)
            L_max: Maximum labor to display (auto-calculated if None)
            save_path: Path to save figure (optional)
        """
        if t is None:
            t = self.params.t
        
        # Calculate key points
        L_star = self.equilibrium_labor()
        w_star = self.equilibrium_wage()
        L_t = self.employment_at_wage_floor(t)
        L_S = self.labor_supplied_at_wage_floor()
        U_t = self.unemployment(t)
        
        # Determine plot range
        if L_max is None:
            L_max = max(L_star * 1.5, L_S * 1.2, 1.0)
        
        L_range = np.linspace(0, L_max, 1000)
        
        # Calculate curves
        w_S_curve = [self.labor_supply(L) for L in L_range]
        w_D_curve = [self.labor_demand(L, t) for L in L_range]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot supply and demand curves
        ax.plot(L_range, w_S_curve, 'b-', linewidth=2, label='Labor Supply')
        ax.plot(L_range, w_D_curve, 'r-', linewidth=2, 
                label=f'Labor Demand (t={t:.2f})')
        
        # Plot wage floor
        ax.axhline(y=self.params.w_bar, color='g', linestyle='--', 
                  linewidth=2, label=f'Wage Floor (w_bar={self.params.w_bar:.2f})')
        
        # Mark initial equilibrium
        ax.plot(L_star, w_star, 'ko', markersize=10, 
               label=f'Initial Equilibrium (L*={L_star:.2f}, w*={w_star:.2f})')
        ax.annotate('(L*, w*)', xy=(L_star, w_star), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='yellow', alpha=0.5))
        
        # Mark employment point
        if L_t > 0:
            ax.plot(L_t, self.params.w_bar, 'ro', markersize=10,
                   label=f'Employment (L(t)={L_t:.2f})')
            ax.annotate('(L(t), w_bar)', xy=(L_t, self.params.w_bar),
                       xytext=(10, -20), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                             facecolor='lightblue', alpha=0.5))
        
        # Mark labor supplied point
        ax.plot(L_S, self.params.w_bar, 'bo', markersize=10,
               label=f'Labor Supplied (L_S={L_S:.2f})')
        ax.annotate('(L_S, w_bar)', xy=(L_S, self.params.w_bar),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                         facecolor='lightgreen', alpha=0.5))
        
        # Shade unemployment
        if U_t > 0 and L_t > 0:
            L_unemp = np.linspace(L_t, L_S, 100)
            ax.fill_between(L_unemp, self.params.w_bar, self.params.w_bar,
                           alpha=0.3, color='red', 
                           label=f'Unemployment (U={U_t:.2f})')
        
        # Formatting
        ax.set_xlabel('Labor (L)', fontsize=12)
        ax.set_ylabel('Wage (w)', fontsize=12)
        ax.set_title(f'Labor Market with Binding Wage Floor (t={t:.2f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(0, L_max)
        ax.set_ylim(0, max(self.params.w_bar * 1.1, w_star * 1.2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, ax
    
    def plot_dynamics(self, num_points: int = 50,
                     save_path: Optional[str] = None):
        """
        Plot how employment and unemployment evolve over time
        
        Args:
            num_points: Number of time points to plot
            save_path: Path to save figure (optional)
        """
        t_range = np.linspace(0, 1, num_points)
        L_t = [self.employment_at_wage_floor(t) for t in t_range]
        U_t = [self.unemployment(t) for t in t_range]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Employment over time
        ax1.plot(t_range, L_t, 'b-', linewidth=2)
        ax1.set_xlabel('Time (t)', fontsize=12)
        ax1.set_ylabel('Employment L(t)', fontsize=12)
        ax1.set_title('Employment Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Unemployment over time
        ax2.plot(t_range, U_t, 'r-', linewidth=2)
        ax2.set_xlabel('Time (t)', fontsize=12)
        ax2.set_ylabel('Unemployment U(t)', fontsize=12)
        ax2.set_title('Unemployment Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, (ax1, ax2)
    
    def summary(self, t: Optional[float] = None) -> str:
        """
        Generate a summary of the model state
        
        Args:
            t: Time index (defaults to self.params.t)
            
        Returns:
            Formatted summary string
        """
        if t is None:
            t = self.params.t
        
        L_star = self.equilibrium_labor()
        w_star = self.equilibrium_wage()
        L_t = self.employment_at_wage_floor(t)
        L_S = self.labor_supplied_at_wage_floor()
        U_t = self.unemployment(t)
        dL_dt = self.employment_derivative()
        
        summary = f"""
Labor Market Model Summary
{'=' * 50}
Parameters:
  Supply intercept (a_S): {self.params.a_S:.4f}
  Supply slope (b_S): {self.params.b_S:.4f}
  Demand intercept at t=0 (a_D0): {self.params.a_D0:.4f}
  Demand slope (b_D): {self.params.b_D:.4f}
  Demand shift magnitude (k): {self.params.k:.4f}
  Wage floor (w_bar): {self.params.w_bar:.4f}
  Time index (t): {t:.4f}

Initial Equilibrium (Pre-Wage Floor):
  Equilibrium labor (L*): {L_star:.4f}
  Equilibrium wage (w*): {w_star:.4f}

At Time t = {t:.4f}:
  Demand intercept (a_D(t)): {self.labor_demand_intercept(t):.4f}
  Employment (L(t)): {L_t:.4f}
  Labor supplied (L_S): {L_S:.4f}
  Unemployment (U(t)): {U_t:.4f}
  Employment derivative (dL/dt): {dL_dt:.4f}

Comparative Statics:
  Employment change rate: {dL_dt:.4f} (negative for k>0)
  {'Employment declining' if dL_dt < 0 else 'Employment constant' if dL_dt == 0 else 'Employment increasing'}
  {'Unemployment increasing' if U_t > 0 and dL_dt < 0 else 'No unemployment' if U_t == 0 else 'Unemployment present'}
{'=' * 50}
"""
        return summary
