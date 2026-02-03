# Labor Market Model with Binding Wage Floor

This Python implementation models a perfectly competitive labor market with a binding wage floor and dynamic adjustment over time.

## Model Description

The model assumes:
- Labor markets are perfectly competitive
- Labor supply is upward sloping and fixed over time
- A binding wage floor $\bar{w}$ is imposed such that $\bar{w} > w^*$
- Over time, firms adjust to the wage floor through capital substitution, automation, or exit, modeled as an inward shift of labor demand

## Key Equations

### Labor Supply
$$w_S(L) = a_S + b_S L$$

### Labor Demand
$$a_D(t) = a_{D0} - k t$$
$$w_D(L,t) = a_D(t) - b_D L$$

### Initial Competitive Equilibrium
$$L^* = \frac{a_{D0} - a_S}{b_S + b_D}$$
$$w^* = a_S + b_S L^*$$

### Employment Under Wage Floor
$$L(t) = \max\left\{0, \frac{a_D(t) - \bar{w}}{b_D}\right\}$$

### Unemployment
$$U(t) = \max\left\{0, L_S(\bar{w}) - L(t)\right\}$$

where $L_S(\bar{w}) = \frac{\bar{w} - a_S}{b_S}$

## Usage

### Basic Example

```python
from labor_market_model import LaborMarketParams, LaborMarketModel

# Define parameters
params = LaborMarketParams(
    a_S=5.0,      # labor supply intercept
    b_S=0.5,      # labor supply slope
    a_D0=20.0,    # labor demand intercept at t=0
    b_D=1.0,      # labor demand slope
    k=3.0,        # magnitude of inward demand shift
    w_bar=12.0,   # binding wage floor
    t=0.0         # time index [0, 1]
)

# Create model
model = LaborMarketModel(params)

# Print summary
print(model.summary())

# Plot market at specific time
model.plot_market(t=0.5)

# Plot dynamics over time
model.plot_dynamics()
```

### Running the Example

```bash
python example_usage.py
```

## Parameters

- `a_S`: Labor supply intercept
- `b_S > 0`: Labor supply slope
- `a_D0`: Labor demand intercept at t=0
- `b_D > 0`: Labor demand slope
- `k ≥ 0`: Magnitude of inward demand shift over time
- `w_bar`: Binding wage floor (must satisfy $\bar{w} > w^*$)
- `t ∈ [0,1]`: Time index (0 = short run, 1 = long run)

## Features

- **Validation**: Automatically checks that the wage floor is binding
- **Visualization**: Generates plots showing:
  - Supply and demand curves
  - Wage floor
  - Initial equilibrium
  - Employment and unemployment at wage floor
  - Dynamics over time
- **Analysis**: Computes employment, unemployment, and comparative statics

## Model Behavior

- Employment declines monotonically over time: $\frac{dL(t)}{dt} = -\frac{k}{b_D} < 0$ for $k>0$
- Unemployment rises monotonically as demand shifts inward
- The wage remains fixed at $\bar{w}$ while employment and unemployment adjust

## Requirements

- Python 3.7+
- numpy
- matplotlib

## Installation

```bash
pip install numpy matplotlib
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## How to Run

### Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the example script**:
   ```bash
   python example_usage.py
   ```

   This will:
   - Display a summary of the model with all parameters and calculations
   - Generate plots at different time points (t=0, t=0.5, t=1.0)
   - Create a dynamics plot showing employment and unemployment over time
   - Save all plots as PNG files in the current directory
   - Display comparative statics showing how employment and unemployment change over time

3. **View the generated plots**:
   - `labor_market_t0.png` - Market state at t=0 (short run)
   - `labor_market_t05.png` - Market state at t=0.5 (intermediate)
   - `labor_market_t1.png` - Market state at t=1.0 (long run)
   - `labor_market_dynamics.png` - Employment and unemployment dynamics over time

### Running Custom Code

You can also use the model in your own Python scripts:

```python
from labor_market_model import LaborMarketParams, LaborMarketModel

# Define your parameters
params = LaborMarketParams(
    a_S=5.0,      # labor supply intercept
    b_S=0.5,      # labor supply slope
    a_D0=20.0,    # labor demand intercept at t=0
    b_D=1.0,      # labor demand slope
    k=3.0,        # magnitude of inward demand shift
    w_bar=12.0,   # binding wage floor (must be > equilibrium wage)
    t=0.0         # time index [0, 1]
)

# Create and use the model
model = LaborMarketModel(params)

# Get summary
print(model.summary())

# Calculate specific values
employment = model.employment_at_wage_floor(t=0.5)
unemployment = model.unemployment(t=0.5)
print(f"At t=0.5: Employment={employment:.2f}, Unemployment={unemployment:.2f}")

# Generate plots
model.plot_market(t=0.5, save_path='my_plot.png')
model.plot_dynamics(save_path='my_dynamics.png')
```

### Expected Output

When you run `example_usage.py`, you should see output like:

```
Labor Market Model Summary
==================================================
Parameters:
  Supply intercept (a_S): 5.0000
  Supply slope (b_S): 0.5000
  ...
  
At Time t = 0.0000:
  Employment (L(t)): 8.0000
  Labor supplied (L_S): 14.0000
  Unemployment (U(t)): 6.0000
  ...

Generating plot for t=0 (short run)...
Figure saved to labor_market_t0.png
...

Comparative Statics Over Time
============================================================
t=0.00: Employment=8.0000, Unemployment=6.0000
t=0.25: Employment=7.2500, Unemployment=6.7500
...
```
