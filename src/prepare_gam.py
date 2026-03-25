import numpy as np
import pandas as pd
import sys
import src.utils as utils
from sklearn.linear_model import LogisticRegression
import time
import pickle

import rpy2
from rpy2.robjects.packages import importr
from rpy2 import robjects
import rpy2.robjects.numpy2ri
# Set up numpy conversion
# Note: activate() is deprecated and raises an exception in newer rpy2 versions
# We use localconverter context manager in functions that need numpy conversion
# For global setup, we try to set up the conversion manually
import warnings
import rpy2.robjects.conversion as cv

# Try to set up numpy conversion globally
# First try the old activate() method (works in older versions)
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        rpy2.robjects.numpy2ri.activate()
except (DeprecationWarning, Exception):
    # activate() failed, try to set up conversion manually using new API
    try:
        # Get the numpy converter and default converter
        numpy_conv = rpy2.robjects.numpy2ri.converter
        default_conv = cv.get_conversion()
        
        # Create a new converter that chains both
        # The numpy converter's py2rpy is a singledispatch function
        # We need to register numpy array conversion with the default converter
        from rpy2.robjects.conversion import Converter
        from functools import singledispatch
        
        # Create a chained converter
        chained = Converter('chained', template=default_conv)
        
        # Register numpy array conversion
        @chained.py2rpy.register(np.ndarray)
        def _numpy_to_r(obj):
            return numpy_conv.py2rpy(obj)
        
        # Set as the global converter
        cv.set_conversion(chained)
    except Exception:
        # If manual setup fails, we'll use localconverter in functions
        pass

def fit_fastsparse(X, y, tmp_lambda0=None, tmp_lambda2=None):
    """
    X, y: numpy arrays. X shape is n*(p+1) and y is either 1 or -1. 
    """
    # Use localconverter with a chained converter that includes both default and numpy conversions
    # This ensures numpy arrays are converted while preserving SexpEnvironment and other default conversions
    from rpy2.robjects.conversion import localconverter, Converter, get_conversion
    
    # Create a chained converter that includes both default and numpy conversions
    # This preserves SexpEnvironment and other default conversions while adding numpy support
    default_conv = get_conversion()
    numpy_conv = rpy2.robjects.numpy2ri.converter
    
    # Create chained converter based on default (preserves SexpEnvironment, etc.)
    chained_conv = Converter('chained', template=default_conv)
    
    # Register numpy array conversion for Python -> R
    @chained_conv.py2rpy.register(np.ndarray)
    def _numpy_to_r(obj):
        return numpy_conv.py2rpy(obj)
    
    # The default converter's rpy2py already handles R -> Python conversion including numpy arrays
    
    with localconverter(chained_conv):
        base = importr('base')
        FastSparse = importr('FastSparse')
        np.random.seed(seed=3337)

        if tmp_lambda0 is None:  
            tmp_lambda0 = 1
        
        if tmp_lambda2 is None:
            tmp_lambda2 = 1e-5

        d = {'package.dependencies': 'package_dot_dependencies', 'package_dependencies': 'package_uscore_dependencies'}
        FastSparse = importr('FastSparse', robject_translations = d)

        fit = FastSparse.FastSparse_fit(X, y, loss="Logistic", penalty="L0L2", intercept=True, algorithm="CDPSI", maxSuppSize = 300, autoLambda=False, lambdaGrid=[tmp_lambda0], nGamma = 1, gammaMin = tmp_lambda2, gammaMax = tmp_lambda2)
   
        betas_fastSparse = base.as_matrix(FastSparse.coef_FastSparse(fit))
        betas_fastSparse = np.asarray(betas_fastSparse)

    return betas_fastSparse

def get_fastsparse(data, lamb0, lamb2, verbose=False):
    X_orig, counts = utils.one_hot_encoding(data.iloc[:,:-1], one_hot=False) # n*p, no intercept column
    y_orig = pd.DataFrame(data.iloc[:,-1]) # {0,1}
    header = list(X_orig.columns)
    header = pd.Index(["intercept"] + header)
    header = header.astype("object")
    if verbose:
        print("header dimension", len(header), flush=True)

    X, y = utils.get_X_y(X_orig, y_orig) # add a column of one to X_orig and make y in {1,-1}

    # Important to reweight the lamb0 before feed into the fastsparse algorithm
    w = fit_fastsparse(X_orig.values, y, tmp_lambda0=lamb0*y.shape[0], tmp_lambda2=lamb2)
    w = w.ravel() # (p+1, ) 
    acc, auc = utils.get_acc_and_auc(w, X, y)
    if verbose:
        print("lamb0:{}, lamb2:{}, acc:{}, auc:{}, supp_size:{}".format(lamb0, lamb2, acc, auc, np.count_nonzero(w)), flush=True)
    
    return w, y, header, X_orig

def prepare_sparse_gam(dname, lamb0, lamb2, multiplier, X_new = None, y = None, header=None, header_new = None, verbose=False):
    """
    multiplier: 1+eps. Rashomon set = {w : loss(w) <= (1+eps)*best_loss}. Stored rset_bound = multiplier*best_loss.
    """
    lamb = 2 * lamb2
    
    if X_new is None:
        # Only read dataset if X_new is not provided
        data = pd.read_csv(f"/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark/{dname}.csv")
        w, y, header = get_fastsparse(data, lamb0, lamb2)
        y = y.ravel()
        X_new, header_new = utils.binary_to_one_hot(data.iloc[:,:-1], w, header)
    
    sample_p = X_new.sum(0)/X_new.shape[0]
    # sample_p[0] = 1e-5
    assert(sample_p.min()!=0)
    X_new_normalized = X_new/np.sqrt(sample_p)
    print("X_new_normalized shape:", X_new_normalized.shape, flush=True)
    print("y shape:", y.shape, flush=True)
    print("lamb:", lamb, flush=True)
    LR_model = LogisticRegression(penalty="l2", C=1/lamb, fit_intercept=False, solver='liblinear', intercept_scaling=10000, max_iter=1000)
    LR_model.fit(X_new_normalized, (y+1)//2) # change y to {0,1}

    w_new_normalized = LR_model.coef_
    w_new = w_new_normalized/np.sqrt(sample_p)
    w_new_normalized = w_new_normalized.ravel()
    w_new = w_new.ravel() # (m+1,) np array
    
    log_loss = utils.get_log_loss(X_new, y, w_new, lamb2, sample_p)
    log_loss_normalized = utils.get_log_loss(X_new_normalized, y, w_new_normalized, lamb2, np.ones(X_new.shape[1]))
    if verbose:
        print('objective:', log_loss, "objective in LR", log_loss_normalized)

    H = utils.hessian(w_new, X_new, y, lamb2, sample_p)
    print("H shape:", H.shape, flush=True)

    outfile = f"/usr/xtmp/vb97/GAM_Rashomon_Sets/gam-rashoman-set/models/{dname}_{lamb0}_{lamb2}_{multiplier}.p"
    # Rashomon set = {w : loss(w) <= (1+eps)*best_loss}. Here multiplier = 1+eps, rset_bound = multiplier*best_loss.
    rset_bound = multiplier * log_loss
    print("multiplier:", multiplier, flush=True)
    if multiplier < 1:
        print("error: multiplier < 1")
        return None
    
    if verbose:
        print("rset_bound (=(1+eps)*best_loss):{}, log objective:{}, multiplier (1+eps):{}".format(rset_bound, log_loss, multiplier))

    results = {
        "date": time.strftime("%d/%m/%y", time.localtime()),
        "data_file": dname,
        "X": X_new,
        "y": y,
        "header_orig": header,
        "header_new": header_new,
        "p": w_new.shape[0], # including intercept, (m+1,)
        "sample_proportion": sample_p, 
        "lamb0": lamb0,
        "lamb2": lamb2,
        "multiplier": multiplier, 
        "rset_bound": rset_bound, 
        "w_orig": w_new, 
        "log_loss_orig": log_loss,
        "hessian": H,
    }
    print("saving model to:", outfile, flush=True)
    with open(outfile, 'wb') as out:
        pickle.dump(results, out, protocol=pickle.DEFAULT_PROTOCOL)
    print("model saved to:", outfile, flush=True)
    return outfile
