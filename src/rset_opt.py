import numpy as np
import pandas as pd
import pickle
import src.utils as utils
import time
import warnings
import torch
import torch.nn as nn
from itertools import combinations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

class RSetOPT:
    def __init__(self, filepath, C=500, lr=0.0001):
        with open(filepath, "rb") as f:
            out = pickle.load(f)
        self.filepath = filepath
        self.dname = out["data_file"]
        data = pd.read_csv("/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark/{}.csv".format(self.dname))
        y = data.iloc[:,-1].values
        if np.min(y) == 0:
            y_max, y_min = np.max(y), np.min(y)
            y = -1 + 2 * (y-y_min)/(y_max-y_min)
        y = y.ravel()
        self.y = y
        self.X = out["X"]
        self.sample_p = out["sample_proportion"]
        self.N = self.X.shape[0]
        self.P = out["p"]
        self.xlabel = utils.get_xlabel(data, out["header_new"])
        self.w_orig = out["w_orig"]
        self.H = out["hessian"]
        self.lamb0 = out["lamb0"]
        self.lamb2 = out["lamb2"]
        self.multiplier = out["multiplier"]  # 1+eps
        self.rset_bound = out["rset_bound"]  # (1+eps)*best_loss; Rashomon set = {w : loss(w) <= rset_bound}
        self.ub = (self.rset_bound / self.multiplier) * (self.multiplier - 1)
        self.C = C

        Sigma, V = np.linalg.eigh(self.H)
        self.Sigma = Sigma
        self.V = V
        # Use float32 for better performance (especially on GPU)
        # Can switch to float64 if precision is critical
        self.dtype = torch.float32
        self.H_half = torch.tensor((np.sqrt(Sigma) * V).T, requires_grad=True, device=device, dtype=self.dtype)
        self.w_center = torch.tensor(self.w_orig, requires_grad=True, device=device, dtype=self.dtype)
        self.optimizer = torch.optim.Adam([self.H_half, self.w_center], lr=lr)
        self.device = device  # Store device for use in other methods
        
        # Cache X, y, and sample_p as tensors to avoid recreating them every iteration
        self.X_torch = torch.tensor(self.X, device=device, dtype=self.dtype)
        self.y_torch = torch.tensor(self.y, device=device, dtype=self.dtype)
        self.sample_p_torch = torch.tensor(self.sample_p, device=device, dtype=self.dtype)

    def get_normalized_H(self):
        """
        the ellipsoid defined by H_half is
        1/2 * (w-w_orig)^T H (w-w_orig) <= ub
        sometime we want to get rid of 1/2 and ub to make it more principled
        """
        return self.H / (2*self.ub)

    def sample_in_ellipsoid(self, n_samples,sample_from_surface=False):
        H, ub = self.H, self.ub
        d = self.P
        u = np.random.normal(size=(n_samples,d)) # randomly sample iid gaussian
        u = u/(np.linalg.norm(u,axis=1).reshape(-1,1)) # normalize to get uniformly random unit vectors
        r = (np.random.random(size=n_samples))**(1/d) # sample radius (uniformly in a sphere)
        if sample_from_surface:
            x_ = u
        else:
            x_ = u * r.reshape(-1,1) # x_ is a uniformly random point in a sphere
        # lamb, V = np.linalg.eigh(H) # eigen decomposition
        a = np.sqrt(2*ub/self.Sigma) # scaling factor
        dw_samples = ((a*self.V) @ x_.T).T # transformation to a ellipsoid
        w_samples = dw_samples + self.w_orig

        return w_samples

    def get_precision(self):
        w_samples = self.sample_in_ellipsoid(10000)
        n_in_rset = 0
        for w_sample in w_samples:
            log_loss = utils.get_log_loss(self.X, self.y, w_sample, self.lamb2, self.sample_p)
            n_in_rset += int(log_loss<=self.rset_bound)
        precision = n_in_rset / w_samples.shape[0]
        print(precision,self.rset_bound)
        return precision

    def get_log_loss_torch(self, w_samples):
        # Use cached tensors instead of recreating them every time
        y_logit = w_samples @ self.X_torch.T
        losses = torch.log(1+torch.exp(-self.y_torch*y_logit))
        # print("avg log loss:{}, std log loss:{}".format(np.mean(loss), np.std(loss)))
        log_losses = losses.mean(1) + self.lamb2 * (self.sample_p_torch[1:] * (w_samples[:,1:]**2)).sum(1)
        return log_losses

    def total_loss(self, w_samples):
        loss_det = self.H_half.det().abs() ** (1/self.P)
        losses_log = self.get_log_loss_torch(w_samples)
        loss_outrset = torch.clamp(losses_log-self.rset_bound,0).mean()
        # print('precision = ', (losses_log<=self.rset_bound).float().mean())

        return loss_det + self.C * loss_outrset

    def sample_in_ellipsoid_torch(self, n_samples=64, sample_from_surface=False):
        d = self.H.shape[0]
        # Use the same device and dtype as H_half and w_center
        u = torch.normal(0, 1, size=(n_samples, d), device=self.device, dtype=self.dtype) # randomly sample iid gaussian
        u = u/(u.norm(dim=1, keepdim=True) + 1e-8) # normalize to get uniformly random unit vectors (add small epsilon for stability)
        r = (torch.rand(size=(n_samples,), device=self.device, dtype=self.dtype))**(1/d) # sample radius (uniformly in a sphere)
        
        if sample_from_surface:
            x_ = u
        else:
            x_ = u * r.reshape(-1,1) # x_ is a uniformly random point in a sphere

        ub = self.ub
        scale = (ub * 2) ** 0.5
        # Regularize H_half to avoid singularity during optimization
        H_reg = self.H_half + 1e-6 * torch.eye(d, device=self.device, dtype=self.dtype)
        out = torch.linalg.solve(H_reg, x_.t())
        dw_samples = (scale * out).t()
        w_samples = dw_samples + self.w_center
        
        return w_samples

    def finetune_ellipsoid(self, n_iters = 1000, verbosity=0):
        if verbosity > 0:
            print('----------- before optimization -----------')
            print('volume proportional to ', 1/self.H_half.det().abs())
            print(f'Using device: {self.device}, dtype: {self.dtype}')
        
        # Use tqdm for progress bar, update with loss periodically
        iter_range = tqdm(range(n_iters), desc="Optimizing ellipsoid") if verbosity == 1 else range(n_iters)
        
        for i in iter_range:
            self.optimizer.zero_grad()
            w_samples = self.sample_in_ellipsoid_torch()
            loss = self.total_loss(w_samples)
            if verbosity > 0:
                print(i, loss.item(), flush=True)
            elif verbosity == 1 and i % 100 == 0:
                # Update progress bar with loss every 100 iterations
                iter_range.set_postfix({'loss': f'{loss.item():.4f}'})
            loss.backward()
            self.optimizer.step()

        if verbosity > 0:
            print('----------- after optimization -----------')
            print('volume proportional to ', 1/self.H_half.det().abs())
        # Convert to numpy, handling both CPU and CUDA tensors
        self.H = (self.H_half.T @ self.H_half).detach().cpu().numpy()
        self.w_orig = self.w_center.detach().cpu().numpy()

    def update_file(self, H_new, w_new):
        with open(self.filepath, "rb") as f:
            res = pickle.load(f)
        res["H_opt"] = H_new
        res["w_opt"] = w_new
        with open(self.filepath, 'wb') as f:
            pickle.dump(res, f, protocol=pickle.DEFAULT_PROTOCOL)