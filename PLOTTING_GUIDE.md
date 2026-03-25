# Plotting Guide for MCMC GAM Output

This guide shows how to plot GAMs from the MCMC method output.

## Key Files

- **Plotting utilities**: `gam_rs_utils/utils.py`
  - `Plotter` class: Simple plotting functions
  - `MultiPlotter` class: Advanced multi-method plotting

- **MCMC output**: The `run_dataset()` method returns a `Results` object with:
  - `w_rset`: Array of shape `(n_samples, n_features)` - all models in Rashomon set
  - `w_opt`: Array of shape `(n_features,)` - optimal model
  - `bin_header`: Header for binned features (use for plotting)

## Basic Usage

### 1. Plot GAM Shape Functions (Simple)

```python
from gam_rs_utils.utils import Plotter
import numpy as np

# After running MCMC
out = mcmc_method.run_dataset(dn='compas', l0=0.01, l2=0.01, eps=0.01, ...)

# Get the data
w_rset = out.w_rset  # All models: (n_samples, n_features)
w_opt = out.w_opt    # Optimal model: (n_features,)
bin_header = out.bin_header  # Feature names

# Plot shape functions for all models
Plotter.plot_gam(bin_header, w_rset)
```

### 2. Plot GAM Shape Functions (MultiPlotter - Recommended)

```python
from gam_rs_utils.utils import MultiPlotter
import numpy as np

# After running MCMC
out = mcmc_method.run_dataset(dn='compas', l0=0.01, l2=0.01, eps=0.01, ...)

# Create plotter
plotter = MultiPlotter(title="GAM Shape Functions - MCMC")

# Add shape functions (exclude intercept column)
plotter.add_shape_function(
    header=np.array(bin_header[1:]),  # Skip intercept
    w_rset=out.w_rset[:, 1:],         # Skip intercept column
    method_name="MCMC"
)

# Plot
fig = plotter.plot_shape_functions()
plt.show()
# Or save: fig.savefig('mcmc_shape_functions.png', dpi=300, bbox_inches='tight')
```

### 3. Compare Multiple Methods

```python
from gam_rs_utils.utils import MultiPlotter
import numpy as np

# Run different methods
mcmc_out = mcmc_method.run_dataset(...)
ellipsoid_out = ellipsoid_method.run_dataset(...)
swapping_out = swapping_method.run_dataset(...)

# Create plotter
plotter = MultiPlotter(title="GAM Shape Functions Comparison")

# Add each method
plotter.add_shape_function(
    header=np.array(mcmc_out.bin_header[1:]),
    w_rset=mcmc_out.w_rset[:, 1:],
    method_name="MCMC"
)

plotter.add_shape_function(
    header=np.array(ellipsoid_out.bin_header[1:]),
    w_rset=ellipsoid_out.w_rset[:, 1:],
    method_name="Ellipsoid"
)

plotter.add_shape_function(
    header=np.array(swapping_out.bin_header[1:]),
    w_rset=swapping_out.w_rset[:, 1:],
    method_name="Swapping"
)

# Plot
fig = plotter.plot_shape_functions()
plt.show()
```

### 4. Plot Loss Distributions

```python
from gam_rs_utils.utils import MultiPlotter
from gam_rs_utils.utils import get_log_loss

# Compute losses for all models
losses = []
for w in out.w_rset:
    loss = get_log_loss(bin_X, y, w, l2, sample_p)
    losses.append(loss)

opt_loss = get_log_loss(bin_X, y, out.w_opt, l2, sample_p)

# Plot
plotter = MultiPlotter(title="Loss Distributions")
plotter.add_distribution(losses, "MCMC", opt_loss=opt_loss)
fig = plotter.plot_distributions()
plt.show()
```

### 5. Plot Variable Importance

```python
from gam_rs_utils.utils import Plotter, MultiPlotter

# Get variable importance
feature_to_vi = Plotter.get_variable_importance(
    bin_X, 
    out.w_rset, 
    np.array(bin_header)
)

# Plot individual histograms
Plotter.plot_variable_importance(feature_to_vi)

# Or use MultiPlotter for comparison
plotter = MultiPlotter(title="Variable Importance Distributions")
plotter.add_variable_importance_distribution(feature_to_vi, "MCMC")
fig = plotter.plot_variable_importance_distributions()
plt.show()
```

## Complete Example

```python
import sys
sys.path.append('../')
from method_scripts import mcmc
from gam_rs_utils.utils import MultiPlotter, Plotter
import numpy as np
import matplotlib.pyplot as plt

# Run MCMC
mcmc_method = mcmc.MCMCMethod()
out = mcmc_method.run_dataset(
    dn='compas', 
    l0=0.01, 
    l2=0.01, 
    eps=0.01,
    proposal_function='swap',
    sigma2=10.0,
    beta=1.0,
    mh_variant='standard'
)

# Plot shape functions
plotter = MultiPlotter(title="MCMC GAM Shape Functions")
plotter.add_shape_function(
    header=np.array(out.bin_header[1:]),  # Skip intercept
    w_rset=out.w_rset[:, 1:],              # Skip intercept column
    method_name="MCMC"
)
fig = plotter.plot_shape_functions()
plt.show()

# Save figure
fig.savefig('mcmc_gam_shapes.png', dpi=300, bbox_inches='tight')
```

## Available Plotting Functions

### Plotter (Simple plots)
- `plot_gam(header, list_of_weights)` - Plot GAM step functions
- `plot_distribution(losses, opt_loss)` - Plot loss histogram
- `plot_variable_importance(feature_to_vi)` - Plot VI histograms
- `get_variable_importance(X, betas, header)` - Compute VI

### MultiPlotter (Advanced multi-method plots)
- `add_shape_function(header, w_rset, method_name)` - Add shape functions
- `plot_shape_functions()` - Plot all shape functions
- `add_distribution(losses, method_name, opt_loss)` - Add loss distribution
- `plot_distributions()` - Plot all distributions
- `add_variable_importance_distribution(feature_to_vi, method_name)` - Add VI
- `plot_variable_importance_distributions()` - Plot all VI distributions
- `add_shape_diversity(feature_to_diversity, method_name)` - Add diversity
- `plot_shape_diversity()` - Plot shape diversity

## Notes

- **Header format**: Use `bin_header[1:]` to skip the intercept
- **Weights format**: Use `w_rset[:, 1:]` to skip the intercept column
- **Shape functions**: Show how each feature's contribution varies across models
- **Loss distributions**: Show the spread of model quality
- **Variable importance**: Show which features are most important across models

