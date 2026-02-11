# GAM Rashomon Set Methods

This directory contains the implementation of several methods for finding models in the GAM Rashomon set.

While multiple algorithms are implemented, **active development and testing currently focus on the `EllipsoidMethod`, `MCMC`-based sampling, and `SwappingMethod`**. The `BlockingMethod`, `QuadraticMethod`, and `Hybrid`-style approaches are **kept for backward compatibility and comparison only** and should be considered **deprecated** for new experiments.

## Class Structure

### Base Class: `BaseGAMRSetMethod`

The `BaseGAMRSetMethod` class provides:
- Common interface for all methods
- Shared functionality for dataset processing
- Result creation and saving
- Progress reporting

### Method Classes

Each method inherits from `BaseGAMRSetMethod` and implements its specific logic:

1. **`SwappingMethod`** - Uses FasterRisk's swapping algorithm (actively used)
2. **`EllipsoidMethod`** - Uses various ellipsoid sampling strategies (primary focus)
3. **`MCMC` / `MCMCMethod`** - Uses Markov chain Monte Carlo to sample from (or near) the Rashomon set (primary focus)
4. **`BlockingMethod`** *(deprecated)* - Legacy blocking-style optimization, kept for reference
5. **`QuadraticMethod`** *(deprecated)* - Legacy extremal sampling with quadratic programming, kept for reference
6. **`Hybrid` / related scripts** *(deprecated)* - Historical experimentation combining multiple strategies

## Usage

### Running Individual Methods

Each method can be run independently. In practice, **most experiments now use `swapping`, `ellipsoid`, or `mcmc`**, and the blocking / quadratic / hybrid variants are only run for ablation or historical comparison.

```bash
# Run swapping method
python swapping_method.py

# Run ellipsoid method
python ellipsoid_method.py

# Run MCMC method
python mcmc.py

# (Deprecated) Run blocking method
python blocking_method.py

# (Deprecated) Run quadratic method
python quadratic_method.py
```

### Using the Runner Script

Use the unified runner script to execute any method:

```bash
python run_method.py <method_name>
```

Available methods: `swapping`, `ellipsoid`, `mcmc`, `blocking` (deprecated), `quadratic` (deprecated)

Example:
```bash
python run_method.py swapping
```

## Method Details

### Swapping Method
- Uses FasterRisk's `RiskScoreOptimizer`
- Performs beam search with feature swapping
- Finds diverse models through iterative optimization

### Ellipsoid Method
- Implements multiple sampling strategies on ellipsoid surface
- Methods include: uniform, poisson, and permutation sampling
- Supports both Euclidean and non-Euclidean distance metrics

### MCMC Method
- Uses MCMC to explore the Rashomon set (or its approximation)
- Produces a sample of GAM models for downstream diversity / stability analysis
- Often paired with utilities in `gam_rs_utils/utils.py` for evaluation and plotting

### Deprecated Methods
- **Blocking Method** *(deprecated)*:
  - Uses blocking optimization to find support sets
  - Merges features into blocks for efficiency
  - Applies hard thresholding to maintain sparsity
- **Quadratic Method** *(deprecated)*:
  - Uses extremal sampling with quadratic programming
  - Finds diverse models by maximizing projections in an ellipsoid
  - Applies hard thresholding to maintain sparsity
- **Hybrid-style methods** *(deprecated)*:
  - Early experimental combinations of the above ideas
  - Kept for historical reference only; not recommended for new experiments

## Benefits of the New Structure

1. **Code Reuse**: Common functionality is shared through the base class
2. **Type Safety**: All methods use proper type hints
3. **Documentation**: Each class and method has comprehensive docstrings
4. **Maintainability**: Changes to common functionality only need to be made in one place
5. **Extensibility**: New methods can easily be added by inheriting from the base class
6. **Consistency**: All methods follow the same interface and patterns

## File Structure

```
method_scripts/
├── base_method.py          # Base class with common functionality
├── swapping_method.py      # Swapping method implementation (actively used)
├── ellipsoid_method.py     # Ellipsoid method implementation (actively used)
├── mcmc.py                 # MCMC-based sampler (actively used)
├── blocking_method.py      # Blocking method implementation (deprecated)
├── quadratic_method.py     # Quadratic method implementation (deprecated)
├── run_method.py          # Unified runner script
├── results_class.py       # Result data classes
└── README.md             # This documentation
```

## Utility Modules

Although not in this directory, two closely related utility files are used heavily by these methods:

- **`gam_rs_utils/utils.py`**:
  - Core helper library for the Rashomon set experiments
  - Provides dataset utilities (`DatasetUtils`) for binarizing datasets, converting between cumulative and binned representations, and preparing inputs for GAM-style models
  - Provides model utilities (`ModelUtils`) for computing losses, logits, predictions, hard-thresholding, summarizing results, and working with headers / support sets
  - Includes plotting helpers (`Plotter`, `MultiPlotter`) to visualize loss distributions, variable importance, GAM shape functions, and comparisons across methods
  - Includes metrics (`Metrics`) for diversity and distance between models (e.g., Hamming distance, inverse IoU, Euclidean / cosine distances, logit variance, etc.)

- **`gam_rs_utils/test_utils.py`**:
  - Pytest-based test suite for `DatasetUtils` and `ModelUtils`
  - Contains small synthetic examples that exercise binarization, header manipulation, and weight expansion logic
  - Serves both as regression tests and as concrete examples of how to call the utilities from experiments

## Legacy Files

The original method scripts are still available for reference:
- `swapping.py`
- `blocking.py`
- `quadratic.py`
- `ellipsoid.py`

These can be used as a reference for the original implementation, but the new class-based structure is recommended for new development. 