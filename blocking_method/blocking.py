import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pickle
import os
from src.rset_app import *
from src.rset_opt import *
from sklearn.linear_model import LogisticRegression
import torch
import random
import time


def optimize_support(filepath, n_support_set, n_combs_max = 100, verbosity=0):
    opt = RSetOPT(filepath)
    opt.get_precision()
    print('sample_p', opt.sample_p, flush=True)
    delta_support = opt.P - n_support_set
    opt.rset_bound += opt.lamb0 * delta_support
    print(f"delta_support = {delta_support}, new rsetbound = {opt.rset_bound}, delta rsetbound = {opt.lamb0 * delta_support}", flush=True)
    opt.optimizer = torch.optim.Adam([opt.H_half, opt.w_center], lr=0.00003)
    s = time.time()
    opt.finetune_ellipsoid()
    first_opt_time = time.time() - s
    opt.get_precision()

    model = RSetGAMs(filepath)
    H_ours = model.H = opt.H
    w_orig_ours = model.w_orig = opt.w_orig
    ub_ours = opt.ub
    merge_ranges = model.get_merge_ranges(n_support_set=n_support_set)
    random.shuffle(merge_ranges)
    feature_comb = []
    time_block = []
    precisions_block = []
    volumes_block = []
    hessian_block = []
    w_center_block = []
    ub_block = []

    for index_ranges in merge_ranges:
        print(index_ranges, flush=True)
        s = time.time()
        H_new, w_center_new, ub_new = model.merge_bins(H_ours, w_orig_ours, ub_ours, index_ranges)
        blocking_time = time.time() - s
        if verbosity > 0:
            print("ub_new", ub_new)
        if ub_new > 0.0005:
            volume_block =  1/np.sqrt(abs(np.linalg.det(H_new/(2*ub_new))))
            if verbosity > 0:
                print("volume after blocking ", volume_block)

            # get X_new
            X_new = model.X.copy()
            deleted_indices = set()
            for start_index, end_index in index_ranges:
                X_new[:,start_index] = X_new[:, start_index:end_index+1].sum(1)
                for index in range(start_index+1, end_index+1):
                    deleted_indices.add(index)
            kept_indices = [i for i in range(model.P) if i not in deleted_indices]
            X_new = X_new[:, kept_indices]  
            if verbosity > 0:
                print(X_new.shape, X_new.min(), X_new.max())
            sample_p = X_new.sum(0)/X_new.shape[0]
            assert(sample_p.min()!=0)
            X_new_normalized = X_new/np.sqrt(sample_p)

            # init opt
            opt.X = X_new
            opt.sample_p = sample_p 
            # print('sample_p', opt.sample_p)
            opt.P = X_new.shape[1]  
            ##############################
            opt.H = H_new
            opt.w_orig = w_center_new
            opt.ub = ub_new
            precision_block = opt.get_precision()
            
            # save results
            feature_comb.append(index_ranges)
            time_block.append(blocking_time)
            precisions_block.append(precision_block)
            volumes_block.append(volume_block)
            hessian_block.append(H_new)
            w_center_block.append(w_center_new)
            ub_block.append(ub_new)
            if verbosity > 0:
                print(f'precision block = {precision_block}, volume_block = {volume_block}')
        
        if len(precisions_block)>=n_combs_max:
            break
    
    print(f'size of results = {len(precisions_block)}')
    res = {"indices": feature_comb, 
           "time_first_opt": first_opt_time,
           "time_block": time_block,
           "precisions_block": precisions_block,
           "volumes_block": volumes_block,
           "hessian_block": hessian_block,
           "w_center_block": w_center_block,
           "ub_block": ub_block
        }

    outfile = f"models/{opt.dname}_{opt.lamb0}_{opt.lamb2}_{opt.multiplier}_{opt.binned}_merge_bins_{n_support_set}.p"
    with open(outfile, 'wb') as out:
        pickle.dump(res, out, protocol=pickle.DEFAULT_PROTOCOL)

    return outfile