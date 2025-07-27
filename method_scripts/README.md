# GAM Rashomon Set Methods

This directory contains the implementation of four different methods for finding models in the GAM Rashomon set.

## Class Structure

### Base Class: `BaseGAMRSetMethod`

The `BaseGAMRSetMethod` class provides:
- Common interface for all methods
- Shared functionality for dataset processing
- Result creation and saving
- Progress reporting

### Method Classes

Each method inherits from `BaseGAMRSetMethod` and implements its specific logic:

1. **`SwappingMethod`** - Uses FasterRisk's swapping algorithm
2. **`BlockingMethod`** - Uses blocking optimization
3. **`QuadraticMethod`** - Uses extremal sampling with quadratic programming
4. **`EllipsoidMethod`** - Uses various ellipsoid sampling strategies

## Usage

### Running Individual Methods

Each method can be run independently:

```bash
# Run swapping method
python swapping_method.py

# Run blocking method
python blocking_method.py

# Run quadratic method
python quadratic_method.py

# Run ellipsoid method
python ellipsoid_method.py
```

### Using the Runner Script

Use the unified runner script to execute any method:

```bash
python run_method.py <method_name>
```

Available methods: `swapping`, `blocking`, `quadratic`, `ellipsoid`

Example:
```bash
python run_method.py swapping
```

## Method Details

### Swapping Method
- Uses FasterRisk's `RiskScoreOptimizer`
- Performs beam search with feature swapping
- Finds diverse models through iterative optimization

### Blocking Method
- Uses blocking optimization to find support sets
- Merges features into blocks for efficiency
- Applies hard thresholding to maintain sparsity

### Quadratic Method
- Uses extremal sampling with quadratic programming
- Finds diverse models by maximizing projections in ellipsoid
- Applies hard thresholding to maintain sparsity

### Ellipsoid Method
- Implements multiple sampling strategies on ellipsoid surface
- Methods include: uniform, poisson, and permutation sampling
- Supports both Euclidean and non-Euclidean distance metrics

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
├── swapping_method.py      # Swapping method implementation
├── blocking_method.py      # Blocking method implementation
├── quadratic_method.py     # Quadratic method implementation
├── ellipsoid_method.py    # Ellipsoid method implementation
├── run_method.py          # Unified runner script
├── results_class.py       # Result data classes
└── README.md             # This documentation
```

## Legacy Files

The original method scripts are still available for reference:
- `swapping.py`
- `blocking.py`
- `quadratic.py`
- `ellipsoid.py`

These can be used as a reference for the original implementation, but the new class-based structure is recommended for new development. 