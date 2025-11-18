#!/usr/bin/env python3
"""
Runner script for GAM Rashomon Set methods.

Usage:
    python run_method.py <method_name>
    python run_method.py clear_results

Available methods:
    - ellipsoid
    - blocking
    - quadratic
    - swapping
    - hybrid
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gam_rs_utils.utils import dataset_settings, DATASET_NAMES
from swapping import SwappingMethod
from blocking_method import BlockingMethod
from quadratic import QuadraticMethod
from ellipsoid import EllipsoidMethod
from hybrid import HybridMethod
import shutil

def main():
    """Main function to run the specified method."""
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    method_name = sys.argv[1].lower()
    
    # Create method instance based on argument
    if method_name == "swapping":
        method = SwappingMethod()
    elif method_name == "blocking":
        method = BlockingMethod()
    elif method_name == "quadratic":
        method = QuadraticMethod()
    elif method_name == "ellipsoid":
        method = EllipsoidMethod()
    elif method_name == "hybrid":
        method = HybridMethod()
    elif method_name == "clear_results":
        for dn in DATASET_NAMES:
            shutil.rmtree(f"../results/{dn}/method_results")
    else:
        print(f"Unknown method: {method_name}")
        print(__doc__)
        sys.exit(1)
    
    # Run the method
    print(f"Running {method_name} method...")
    method.run_all_datasets(dataset_settings)
    print(f"Completed {method_name} method.")


if __name__ == "__main__":
    main() 