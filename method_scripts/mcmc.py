import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle as pkl
import random
import inspect
from functools import partial
from typing import Dict, Any, List, Tuple
from src.prepare_gam import *
from src.rset_opt import *
from src.run_app import *
from gam_rs_utils.utils import *
from method_scripts.base_method import BaseGAMRSetMethod
from method_scripts.results import MethodType, Results
from time import time
import itertools
from tqdm import tqdm

INTERCEPT_IDX = 0

class MCMCMethod(BaseGAMRSetMethod):
    """
    MCMC sampling method for finding models in the GAM Rashomon set.
    
    This method uses MCMC sampling to find diverse models in the Rashomon set.
    """
    
    def __init__(self):
        """Initialize the MCMC method."""
        super().__init__(MethodType.MCMC)
        extra = {
            "proposal_function": [
                # "random",
                # "swap", 
                # "correlation_swap", 
                "multi_swap", 
                # "same_feature_swap"
            ],
            "sigma2": [10.0],
            "sample_from_rset": [0, 20],
        }
        extra_settings = []
        for settings in itertools.product(*extra.values()):
            extra_settings.append(dict(zip(extra.keys(), settings)))
        self.extra_settings = extra_settings

    # ---------- MAP logistic regression + Laplace score ---------- 
    @staticmethod
    def map_logistic_theta(X_S, y, sigma2, max_iter=1000):
        # C is inverse of reg strength; C ≈ sigma2 is a rough mapping
        clf = LogisticRegression(
            penalty="l2",
            C=sigma2,
            fit_intercept=False,
            solver="lbfgs",
            max_iter=max_iter,
        )
        clf.fit(X_S, y)
        theta_hat = clf.coef_.ravel()
        proba = clf.predict_proba(X_S)[:, 1]
        return theta_hat, proba

    @staticmethod
    def hessian_and_logdet(X_S, proba, sigma2, jitter=1e-6):
        # proba = π_i = P(y_i=1)
        p = np.clip(proba, 1e-8, 1.0 - 1e-8)
        W = p * (1.0 - p)          # (n,)
        XTW = X_S.T * W            # each row scaled by W
        A = XTW @ X_S + (1.0 / sigma2) * np.eye(X_S.shape[1])

        # add a tiny jitter on top of the prior to ensure positive definiteness
        A += jitter * np.eye(A.shape[0])

        # compute log det via Cholesky if possible
        try:
            L = np.linalg.cholesky(A)      # A = L L^T
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            print("np.linalg.LinAlgError")
            # fallback to slogdet
            sign, logdet = np.linalg.slogdet(A)
            if sign <= 0:
                # last resort: just make it something finite
                logdet = np.log(np.abs(np.linalg.det(A)) + 1e-12)

        return A, logdet

    @staticmethod
    def evaluate_score_laplace(S, X, y, sigma2, p_feat, K, max_iter=1000):
        """
        J(S) ≈ log P(S) + log P(D | S, θ̂_S) + log P(θ̂_S) - 0.5 log det(A_S)
        """

        idx = np.array(sorted(list(S)), dtype=int)
        X_S = X[:, idx]
        n, d = X_S.shape

        # 1. MAP θ and probabilities
        theta_hat, proba = MCMCMethod.map_logistic_theta(X_S, y, sigma2=sigma2, max_iter=max_iter)

        # 2. log-likelihood at θ̂
        eps = 1e-12
        log_likelihood = np.sum(
            y * np.log(proba + eps) + (1.0 - y) * np.log(1.0 - proba + eps)
        )

        # 3. log prior on θ (Gaussian N(0, σ² I), constants dropped)
        log_prior_theta = -0.5 * np.dot(theta_hat, theta_hat) / sigma2

        # 4. Hessian and logdet for Laplace Occam factor
        _, logdet = MCMCMethod.hessian_and_logdet(X_S, proba, sigma2=sigma2)

        # 5. Laplace log-marginal (ignoring constants that cancel in MH)
        #    L(S) ≈ log_likelihood + log_prior_theta - 0.5 * logdet
        L_S = log_likelihood + log_prior_theta - 0.5 * logdet

        # 6. sparsity prior on S: Bernoulli(p_feat) per feature
        #    log P(S) = |S| log p_feat + (K - |S|) log(1 - p_feat)
        k = len(S)
        log_prior_S = k * np.log(p_feat) + (K - k) * np.log(1.0 - p_feat)

        # 7. total score
        J_S = L_S + log_prior_S

        return J_S

    # ---------- Proposal Functions ---------- 
    @staticmethod
    def propose_swap(S, K, score, add_power=1.0, drop_power=1.0, eps=1e-8):
        """
        one-step proposal that preserves support size:
            - choose one non-intercept feature in S to remove
            - choose one feature outside S to add

        S            : current support (set of indices)
        K            : total number of features (0..K-1)
        """

        current = set(S)

        # all non-intercept indices
        all_features = set(range(K))
        all_features.discard(INTERCEPT_IDX)

        in_set  = [j for j in current if j != INTERCEPT_IDX]
        out_set = list(all_features - current)

        if len(in_set) == 0 or len(out_set) == 0:
            # degenerate: nothing to swap, return same support
            return current, 0.0, 0.0

        # pick one to drop, one to add
        f_remove = random.choice(in_set)
        f_add    = random.choice(out_set)

        S_prop = set(current)
        S_prop.remove(f_remove)
        S_prop.add(f_add)

        return S_prop, 0, 0

    @staticmethod
    def compute_corr_score(X, y01, eps=1e-12):
        """
        X   : numpy array (n, K), sparse-in-value but dense ndarray
        y01 : numpy array (n,), values in {0,1}
        returns:
        score : numpy array (K,), higher = more important
        """
        X = np.asarray(X, dtype=float)
        y01 = np.asarray(y01, dtype=float).ravel()

        n, K = X.shape

        # center y
        yc = y01 - y01.mean()
        y_norm = np.sqrt(np.sum(yc * yc)) + eps

        # center X columnwise
        X_mean = X.mean(axis=0)
        Xc = X - X_mean

        # numerator: X^T y
        Xt_y = Xc.T @ yc

        # denominator: ||X_j|| * ||y||
        X_norm = np.sqrt(np.sum(Xc * Xc, axis=0)) + eps

        score = np.abs(Xt_y / (X_norm * y_norm))

        # never bias intercept
        score[INTERCEPT_IDX] = 0.0
        return score

    @staticmethod
    def _normalize_weights(w):
        w = np.asarray(w, dtype=float)
        w = np.clip(w, 0.0, np.inf)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return np.ones_like(w) / len(w)
        return w / s

    @staticmethod
    def propose_weighted_swap(S, K, score, add_power=1.0, drop_power=1.0, eps=1e-12):
        """
        weighted 1-swap proposal:
        - drop one in-support (non-intercept) feature, biased toward low score
        - add one out-of-support (non-intercept) feature, biased toward high score

        returns:
        S_prop, log_q_forward, log_q_backward

        note: non-symmetric in general, so include (log_q_backward - log_q_forward) in MH
        """
        S = set(S)
        score = np.asarray(score, dtype=float)
        if score.shape[0] != K:
            raise ValueError(f"score must have length {K}, got {score.shape[0]}")

        # candidates from current state
        in_set = [j for j in S if j != INTERCEPT_IDX]
        if len(in_set) == 0:
            return S, 0.0, 0.0

        out_set = [j for j in range(K) if (j not in S) and (j != INTERCEPT_IDX)]
        if len(out_set) == 0:
            return S, 0.0, 0.0

        # forward drop probs: favor low score
        s_in = score[in_set]
        w_drop_fwd = (s_in + eps) ** (-drop_power)
        p_drop_fwd = MCMCMethod._normalize_weights(w_drop_fwd)

        # forward add probs: favor high score
        s_out = score[out_set]
        w_add_fwd = (s_out + eps) ** (add_power)
        p_add_fwd = MCMCMethod._normalize_weights(w_add_fwd)

        # sample move
        rng = np.random.default_rng()
        i_drop = rng.choice(len(in_set), p=p_drop_fwd)
        i_add = rng.choice(len(out_set), p=p_add_fwd)
        f_remove = in_set[i_drop]
        f_add = out_set[i_add]

        S_prop = set(S)
        S_prop.remove(f_remove)
        S_prop.add(f_add)

        log_q_forward = float(np.log(p_drop_fwd[i_drop]) + np.log(p_add_fwd[i_add]))

        # compute backward q: q(S | S_prop)
        # from S_prop, we'd need to remove f_add and add f_remove
        in_set_bwd = [j for j in S_prop if j != INTERCEPT_IDX]
        out_set_bwd = [j for j in range(K) if (j not in S_prop) and (j != INTERCEPT_IDX)]

        # sanity: required elements should exist
        if f_add not in in_set_bwd or f_remove not in out_set_bwd:
            log_q_backward = -np.inf
            return S_prop, log_q_forward, log_q_backward

        # backward drop probs (drop from S_prop): favors low score
        s_in_bwd = score[in_set_bwd]
        w_drop_bwd = (s_in_bwd + eps) ** (-drop_power)
        p_drop_bwd = MCMCMethod._normalize_weights(w_drop_bwd)

        # backward add probs (add to S_prop): favors high score
        s_out_bwd = score[out_set_bwd]
        w_add_bwd = (s_out_bwd + eps) ** (add_power)
        p_add_bwd = MCMCMethod._normalize_weights(w_add_bwd)

        i_drop_bwd = in_set_bwd.index(f_add)
        i_add_bwd = out_set_bwd.index(f_remove)

        log_q_backward = float(np.log(p_drop_bwd[i_drop_bwd]) + np.log(p_add_bwd[i_add_bwd]))

        return S_prop, log_q_forward, log_q_backward

    @staticmethod
    def propose_multi_swap(S, K, score, n_swaps=2):
        """
        Multi-swap proposal that preserves support size:
        - randomly swap n_swaps features at once
        
        Args:
            S: current support (set of indices)
            K: total number of features (0..K-1)
            score: feature importance scores (unused for random version)
            n_swaps: number of features to swap simultaneously
            add_power, drop_power: unused, kept for interface consistency
            eps: unused, kept for interface consistency
            
        Returns:
            S_prop, log_q_forward, log_q_backward
        """
        current = set(S)
        
        # all non-intercept indices
        all_features = set(range(K))
        all_features.discard(INTERCEPT_IDX)
        
        in_set = [j for j in current if j != INTERCEPT_IDX]
        out_set = list(all_features - current)
        
        # Ensure we can actually swap
        n_swaps = min(n_swaps, len(in_set), len(out_set))
        if n_swaps == 0 or len(in_set) == 0 or len(out_set) == 0:
            return current, 0.0, 0.0
        
        # Sample features to remove and add
        features_remove = random.sample(in_set, n_swaps)
        features_add = random.sample(out_set, n_swaps)
        
        S_prop = set(current)
        for f in features_remove:
            S_prop.remove(f)
        for f in features_add:
            S_prop.add(f)
        
        # For symmetric proposal, log probabilities are equal
        # Forward: choose n_swaps from |in_set| and n_swaps from |out_set|
        # Backward: choose n_swaps from |S_prop| and n_swaps from complement
        log_q_forward = 0.0  # Uniform random, so log prob is constant (cancels in ratio)
        log_q_backward = 0.0
        
        return S_prop, log_q_forward, log_q_backward

    @staticmethod
    def propose_weighted_multi_swap(S, K, score, n_swaps=2, add_power=1.0, drop_power=1.0, eps=1e-12):
        """
        Weighted multi-swap proposal that preserves support size:
        - swap n_swaps features, weighted by importance scores
        
        Args:
            S: current support (set of indices)
            K: total number of features (0..K-1)
            score: feature importance scores
            n_swaps: number of features to swap simultaneously
            add_power: power for weighting adds (higher = favor high scores)
            drop_power: power for weighting drops (higher = favor low scores)
            eps: small constant for numerical stability
            
        Returns:
            S_prop, log_q_forward, log_q_backward
        """
        S = set(S)
        score = np.asarray(score, dtype=float)
        if score.shape[0] != K:
            raise ValueError(f"score must have length {K}, got {score.shape[0]}")
        
        in_set = [j for j in S if j != INTERCEPT_IDX]
        out_set = [j for j in range(K) if (j not in S) and (j != INTERCEPT_IDX)]
        
        n_swaps = min(n_swaps, len(in_set), len(out_set))
        if n_swaps == 0 or len(in_set) == 0 or len(out_set) == 0:
            return S, 0.0, 0.0
        
        # Forward probabilities
        s_in = score[in_set]
        w_drop_fwd = (s_in + eps) ** (-drop_power)
        p_drop_fwd = MCMCMethod._normalize_weights(w_drop_fwd)
        
        s_out = score[out_set]
        w_add_fwd = (s_out + eps) ** (add_power)
        p_add_fwd = MCMCMethod._normalize_weights(w_add_fwd)
        
        # Sample without replacement
        rng = np.random.default_rng()
        idx_remove = rng.choice(len(in_set), size=n_swaps, replace=False, p=p_drop_fwd)
        idx_add = rng.choice(len(out_set), size=n_swaps, replace=False, p=p_add_fwd)
        
        features_remove = [in_set[i] for i in idx_remove]
        features_add = [out_set[i] for i in idx_add]
        
        S_prop = set(S)
        for f in features_remove:
            S_prop.remove(f)
        for f in features_add:
            S_prop.add(f)
        
        # Compute log probability of forward move (without replacement)
        log_q_forward = 0.0
        p_drop_remaining = p_drop_fwd.copy()
        for i in idx_remove:
            log_q_forward += np.log(p_drop_remaining[i])
            # Remove chosen element and renormalize
            p_drop_remaining = np.delete(p_drop_remaining, i)
            if len(p_drop_remaining) > 0:
                p_drop_remaining = p_drop_remaining / p_drop_remaining.sum()
        
        p_add_remaining = p_add_fwd.copy()
        for i in idx_add:
            log_q_forward += np.log(p_add_remaining[i])
            p_add_remaining = np.delete(p_add_remaining, i)
            if len(p_add_remaining) > 0:
                p_add_remaining = p_add_remaining / p_add_remaining.sum()
        
        # Backward probabilities (from S_prop back to S)
        in_set_bwd = [j for j in S_prop if j != INTERCEPT_IDX]
        out_set_bwd = [j for j in range(K) if (j not in S_prop) and (j != INTERCEPT_IDX)]
        
        # Check if reverse move is possible
        if not all(f in in_set_bwd for f in features_add) or \
           not all(f in out_set_bwd for f in features_remove):
            log_q_backward = -np.inf
            return S_prop, log_q_forward, log_q_backward
        
        s_in_bwd = score[in_set_bwd]
        w_drop_bwd = (s_in_bwd + eps) ** (-drop_power)
        p_drop_bwd = MCMCMethod._normalize_weights(w_drop_bwd)
        
        s_out_bwd = score[out_set_bwd]
        w_add_bwd = (s_out_bwd + eps) ** (add_power)
        p_add_bwd = MCMCMethod._normalize_weights(w_add_bwd)
        
        # Find indices for backward move
        idx_remove_bwd = [in_set_bwd.index(f) for f in features_add]
        idx_add_bwd = [out_set_bwd.index(f) for f in features_remove]
        
        log_q_backward = 0.0
        p_drop_bwd_remaining = p_drop_bwd.copy()
        for i in sorted(idx_remove_bwd, reverse=True):  # Sort reverse to maintain indices
            log_q_backward += np.log(p_drop_bwd_remaining[i])
            p_drop_bwd_remaining = np.delete(p_drop_bwd_remaining, i)
            if len(p_drop_bwd_remaining) > 0:
                p_drop_bwd_remaining = p_drop_bwd_remaining / p_drop_bwd_remaining.sum()
        
        p_add_bwd_remaining = p_add_bwd.copy()
        for i in sorted(idx_add_bwd, reverse=True):
            log_q_backward += np.log(p_add_bwd_remaining[i])
            p_add_bwd_remaining = np.delete(p_add_bwd_remaining, i)
            if len(p_add_bwd_remaining) > 0:
                p_add_bwd_remaining = p_add_bwd_remaining / p_add_bwd_remaining.sum()
        
        return S_prop, log_q_forward, log_q_backward

    @staticmethod
    def propose_add_or_remove(S, K, score, add_power=1.0, drop_power=1.0, eps=1e-8, add_prob=0.5):
        """
        Add-or-remove proposal: randomly decides to add or remove a feature.
        
        Args:
            S: current support (set of indices)
            K: total number of features (0..K-1)
            score: feature importance scores (unused for random version)
            add_power, drop_power: unused, kept for interface consistency
            eps: unused, kept for interface consistency
            add_prob: probability of choosing add (vs remove)
            
        Returns:
            S_prop, log_q_forward, log_q_backward
        """
        current = set(S)
        
        in_set = [j for j in current if j != INTERCEPT_IDX]
        all_features = set(range(K))
        all_features.discard(INTERCEPT_IDX)
        out_set = list(all_features - current)
        
        # Decide whether to add or remove
        if len(in_set) == 0:
            # Can only add
            action = 'add'
        elif len(out_set) == 0:
            # Can only remove
            action = 'remove'
        else:
            action = 'add' if random.random() < add_prob else 'remove'
        
        if action == 'add':
            f_add = random.choice(out_set)
            S_prop = set(current)
            S_prop.add(f_add)
            
            # Forward: add_prob * (1/|out_set|)
            log_q_forward = np.log(add_prob) - np.log(len(out_set))
            
            # Backward: (1 - add_prob) * (1/|in_set_bwd|)
            in_set_bwd = [j for j in S_prop if j != INTERCEPT_IDX]
            if len(in_set_bwd) == 0:
                log_q_backward = -np.inf
            else:
                log_q_backward = np.log(1 - add_prob) - np.log(len(in_set_bwd))
        else:
            f_remove = random.choice(in_set)
            S_prop = set(current)
            S_prop.remove(f_remove)
            
            # Forward: (1 - add_prob) * (1/|in_set|)
            log_q_forward = np.log(1 - add_prob) - np.log(len(in_set))
            
            # Backward: add_prob * (1/|out_set_bwd|)
            out_set_bwd = list(all_features - S_prop)
            if len(out_set_bwd) == 0:
                log_q_backward = -np.inf
            else:
                log_q_backward = np.log(add_prob) - np.log(len(out_set_bwd))
        
        return S_prop, log_q_forward, log_q_backward

    @staticmethod
    def propose_weighted_add_or_remove(S, K, score, add_power=1.0, drop_power=1.0, eps=1e-12, add_prob=0.5):
        """
        Weighted add-or-remove proposal: randomly decides to add or remove,
        with weighted selection based on importance scores.
        
        Args:
            S: current support (set of indices)
            K: total number of features (0..K-1)
            score: feature importance scores
            add_power: power for weighting adds (higher = favor high scores)
            drop_power: power for weighting drops (higher = favor low scores)
            eps: small constant for numerical stability
            add_prob: probability of choosing add (vs remove)
            
        Returns:
            S_prop, log_q_forward, log_q_backward
        """
        S = set(S)
        score = np.asarray(score, dtype=float)
        if score.shape[0] != K:
            raise ValueError(f"score must have length {K}, got {score.shape[0]}")
        
        in_set = [j for j in S if j != INTERCEPT_IDX]
        all_features = set(range(K))
        all_features.discard(INTERCEPT_IDX)
        out_set = [j for j in range(K) if (j not in S) and (j != INTERCEPT_IDX)]
        
        # Decide whether to add or remove
        if len(in_set) == 0:
            action = 'add'
        elif len(out_set) == 0:
            action = 'remove'
        else:
            action = 'add' if random.random() < add_prob else 'remove'
        
        rng = np.random.default_rng()
        
        if action == 'add':
            # Weighted add
            s_out = score[out_set]
            w_add_fwd = (s_out + eps) ** (add_power)
            p_add_fwd = MCMCMethod._normalize_weights(w_add_fwd)
            
            i_add = rng.choice(len(out_set), p=p_add_fwd)
            f_add = out_set[i_add]
            
            S_prop = set(S)
            S_prop.add(f_add)
            
            log_q_forward = np.log(add_prob) + np.log(p_add_fwd[i_add])
            
            # Backward: weighted remove from S_prop
            in_set_bwd = [j for j in S_prop if j != INTERCEPT_IDX]
            if len(in_set_bwd) == 0 or f_add not in in_set_bwd:
                log_q_backward = -np.inf
            else:
                s_in_bwd = score[in_set_bwd]
                w_drop_bwd = (s_in_bwd + eps) ** (-drop_power)
                p_drop_bwd = MCMCMethod._normalize_weights(w_drop_bwd)
                i_drop_bwd = in_set_bwd.index(f_add)
                log_q_backward = np.log(1 - add_prob) + np.log(p_drop_bwd[i_drop_bwd])
        else:
            # Weighted remove
            s_in = score[in_set]
            w_drop_fwd = (s_in + eps) ** (-drop_power)
            p_drop_fwd = MCMCMethod._normalize_weights(w_drop_fwd)
            
            i_drop = rng.choice(len(in_set), p=p_drop_fwd)
            f_remove = in_set[i_drop]
            
            S_prop = set(S)
            S_prop.remove(f_remove)
            
            log_q_forward = np.log(1 - add_prob) + np.log(p_drop_fwd[i_drop])
            
            # Backward: weighted add to complement of S_prop
            out_set_bwd = [j for j in range(K) if (j not in S_prop) and (j != INTERCEPT_IDX)]
            if len(out_set_bwd) == 0 or f_remove not in out_set_bwd:
                log_q_backward = -np.inf
            else:
                s_out_bwd = score[out_set_bwd]
                w_add_bwd = (s_out_bwd + eps) ** (add_power)
                p_add_bwd = MCMCMethod._normalize_weights(w_add_bwd)
                i_add_bwd = out_set_bwd.index(f_remove)
                log_q_backward = np.log(add_prob) + np.log(p_add_bwd[i_add_bwd])
        
        return S_prop, log_q_forward, log_q_backward

    @staticmethod
    def propose_same_feature_swap(S, K, score, header=None):
        """
        one-step proposal that swaps supports of the same feature:
            - choose one feature in S to remove
            - choose a different support in the same feature to add

        S            : current support (set of indices)
        K            : total number of features (0..K-1)
        header       : header of the dataset
        """
        feature_indices = ModelUtils.get_feature_indices(header)
        current = set(S)
        f_remove = random.choice(list(current - {INTERCEPT_IDX}))
        
        for f, indices in feature_indices.items():
            if f_remove in indices:
                in_set = set(indices) - {f_remove}
                break
        
        if len(in_set) == 0:
            return current, 0.0, 0.0
        
        f_add = random.choice(list(in_set))
        S_prop = set(current)
        S_prop.remove(f_remove)
        S_prop.add(f_add)
        return S_prop, 0.0, 0.0

    # ---------- Metropolis-Hastings ---------- 
    @staticmethod
    def mh_sample_supports(X, y, sigma2=10.0, n_steps=2000, p_feat=0.01, beta=0.025, burn_in=500, target_size=30, init_support=None, proposal_fnc=propose_swap) -> Tuple[List[List[int]], List[float]]:
        """
        Metropolis–Hastings sampler over supports of fixed size 'target_size_total'.

        X, y              : data
        sigma2            : prior variance on weights (L2 strength)
        target_size_total : total number of features in support (incl. intercept)
        n_steps           : total MH steps
        p_feat            : expected sparsity
        beta              : MH acceptance ratio parameter
        burn_in           : steps to discard
        init_support      : optional initial support (list/iterable of indices)
        proposal_fnc      : proposal function
        """

        n, K = X.shape

        # ---------- 1. initialize support ----------

        if init_support is not None:
            S_current = set(init_support)
            S_current.add(INTERCEPT_IDX)
        else:
            S_current = {INTERCEPT_IDX}

        # initial score
        J_current_raw = MCMCMethod.evaluate_score_laplace(S_current, X, y, sigma2=sigma2, p_feat=p_feat, K=K)

        samples = []
        scores = []

        accepts = 0
        score = MCMCMethod.compute_corr_score(X, y)

        # ---------- 2. MCMC loop ----------
        skip_by = (n_steps - burn_in) // target_size
        for t in range(n_steps):
            # propose a swap that keeps |S| fixed
            S_prop, log_q_forward, log_q_backward = proposal_fnc(S_current, K, score)

            # evaluate proposed support
            J_prop_raw = MCMCMethod.evaluate_score_laplace(S_prop, X, y, sigma2=sigma2, p_feat=p_feat, K=K)

            # MH acceptance ratio (log scale)
            log_alpha = beta * ((J_prop_raw - J_current_raw) + (log_q_backward - log_q_forward))

            u = np.random.rand()
            if np.log(u) < min(0.0, log_alpha):
                accepts += 1
                S_current = S_prop
                J_current_raw = J_prop_raw

            # store after burn-in
            if t >= burn_in and (t + 1) % skip_by == 0:
                samples.append(sorted(list(set(S_current))))
                scores.append(J_current_raw)
            
            # print progress
            if (t + 1) % skip_by == 0:
                acc_rate = accepts / (t+1)
                print(f"step {t+1}, |S|={len(S_current)}, J={J_current_raw:.2f}, accept_rate={acc_rate:.3f}")

        return samples, scores

    # ---------- Random Sampling ---------- 
    @staticmethod
    def random_sample_supports(X, y, support_size, n_samples, sigma2=10.0, p_feat=0.01) -> Tuple[List[List[int]], List[float]]:
        n, K = X.shape
        samples = []
        scores = []
        for t in range(n_samples):
            S_current = set(random.sample(range(K), support_size - 1))
            S_current.add(INTERCEPT_IDX)
            J_current_raw = MCMCMethod.evaluate_score_laplace(S_current, X, y, sigma2=sigma2, p_feat=p_feat, K=K)
            samples.append(sorted(list(S_current)))
            scores.append(J_current_raw)

            if (t + 1) % (n_samples // 10) == 0:
                print(f"step {t+1}, |S|={len(S_current)}, J={J_current_raw:.2f}")
        return samples, scores

    def run_dataset(self, dn: str, l0: float = None, l2: float = None, 
                      eps: float = None, ne: int = None, n_support_set: int = None,
                      proposal_function: str = "random", sigma2: float = 10.0, r_min: float = None, 
                      sample_from_rset: int = 0, beta: float = 1.0, **kwargs) -> Any:
        """
        Run the MCMC method on a single dataset.
        
        Args:
            dname: Dataset name
            n_samples: Number of samples to generate
            l0: L0 regularization parameter
            l2: L2 regularization parameter
            eps: Epsilon parameter for the rset bound
            num_estimators: Number of estimators
            n_support_set: Number of support features
            proposal_function: Proposal function
            sigma2: Prior variance on weights (L2 strength)
            **kwargs: Additional unused parameters
            
        Returns:
            Results object for this dataset
        """
        TARGET_NUM_SUPPORT_SETS = 50
        N_STEPS = 6000
        BURN_IN = 1000

        # Create or load dataset
        data = pd.read_csv(f"datasets/{dn}.csv")
        fastsparse_data = Results.create_fastsparse_dataset(dn, l0, l2)

        bin_header = fastsparse_data['bin_header']
        bin_X = fastsparse_data['bin_X']

        cum_header = fastsparse_data['cum_header']
        cum_X = fastsparse_data['cum_X']

        y = fastsparse_data['y']
        w = fastsparse_data['w']
        starting_support = w.nonzero()[0].tolist()

        # ---------- 0. prepare data ----------
        X = cum_X.values if hasattr(cum_X, "values") else cum_X
        y_arr01 = y.values.astype(float) if hasattr(y, "values") else y.astype(float)
        y_arr01 = (y_arr01 > 0).astype(int)
        n, K = X.shape

        # Prepare sparse GAM
        def fit_ellipsoid_and_sample(support_set, sample_from_rset):
            t0 = time()
            w = np.zeros(K)
            w[support_set] = 1
            sparse_X, sparse_header = utils.binary_to_one_hot(data.iloc[:,:-1], w, cum_header)
            sparse_gam_file = prepare_sparse_gam(dn, l0, l2, eps, sparse_X, y, cum_header, sparse_header)
            if sparse_gam_file is None:
                return None, None, None, None

            t1 = time()
            model = RSetOPT(sparse_gam_file)
            model.finetune_ellipsoid(verbosity=0)
            H_opt = model.get_normalized_H()
            model.update_file(H_opt, model.w_orig)

            with open(sparse_gam_file, 'rb') as f:
                sparse_gam_data = pickle.load(f)
            w = sparse_gam_data['w_orig']
            m = sparse_gam_data['multiplier']

            header_object = ModelUtils.get_header_object(cum_header)
            sparse_header_object = ModelUtils.get_header_object(sparse_header)
            expanded_w = ModelUtils.expand_w(w, sparse_header_object, header_object)
            w_samples, rset = get_models_from_rset(
                sparse_gam_file, eps, n_samples=sample_from_rset, plot_shape=False, 
                sampling="uniform", distance_metric=None, r_min=r_min,
            )

            t2 = time()
            if sample_from_rset:
                expanded_w_samples = ModelUtils.expand_w_samples(w_samples, sparse_header_object, header_object)
                return [expanded_w_samples, np.array([expanded_w])], m, t1 - t0, t2 - t1
            return [expanded_w], m, t1 - t0, t2 - t1

        mcmc_sampling_and_fitting_time = 0
        ellipsoid_sampling_time = 0

        # get optimal model
        print("fitting optimal model")
        w_opt, m, t0, t1 = fit_ellipsoid_and_sample(starting_support, 0)
        w_opt = w_opt[0]
        mcmc_sampling_and_fitting_time += t0
        ellipsoid_sampling_time += t1

        # MCMC sampling
        t0 = time()
        print("MCMC sampling...")
        support_sets, scores = [], []
        if proposal_function == "random":
            print("random sampling...")
            support_sets, scores = MCMCMethod.random_sample_supports(
                X=X,
                y=y_arr01,
                support_size=len(starting_support),
                n_samples=TARGET_NUM_SUPPORT_SETS,
                sigma2=sigma2,
            )
        else:
            def _bind_proposal(fn, /, **maybe_kwargs):
                """
                Bind only kwargs that the proposal function accepts.

                This lets different proposal functions take different optional params
                without breaking a generic call site: proposal_fnc(S, K, score).
                """
                sig = inspect.signature(fn)
                accepted = {
                    k: v for k, v in maybe_kwargs.items()
                    if k in sig.parameters and v is not None
                }
                return partial(fn, **accepted)

            # Defaults for proposals that need fixed params; user can override via **kwargs
            proposal_defaults = {
                "multi_swap": {"n_swaps": 2},
                "same_feature_swap": {"header": cum_header},
                # "weighted_multi_swap": {"n_swaps": kwargs.get("n_swaps", 2)},
                # "add_or_remove": {"add_prob": kwargs.get("add_prob", 0.5)},
                # "weighted_add_or_remove": {"add_prob": kwargs.get("add_prob", 0.5)},
            }

            # Map proposal function names to base functions
            proposal_map = {
                "swap": MCMCMethod.propose_swap,
                "correlation_swap": MCMCMethod.propose_weighted_swap,
                "multi_swap": MCMCMethod.propose_multi_swap,
                "same_feature_swap": MCMCMethod.propose_same_feature_swap,
                # "weighted_multi_swap": MCMCMethod.propose_weighted_multi_swap,
                # "add_or_remove": MCMCMethod.propose_add_or_remove,
                # "weighted_add_or_remove": MCMCMethod.propose_weighted_add_or_remove,
            }
            
            if proposal_function not in proposal_map:
                raise ValueError(f"Unknown proposal function: {proposal_function}. "
                               f"Available: {list(proposal_map.keys())}")
            
            base_proposal_fnc = proposal_map[proposal_function]
            proposal_fnc = _bind_proposal(
                base_proposal_fnc,
                **proposal_defaults.get(proposal_function, {}),
                **kwargs,  # allow per-proposal optional args (e.g., add_power, drop_power, eps, n_swaps, add_prob)
            )
            print(f"Using proposal function: {proposal_function}")
            
            support_sets, scores = MCMCMethod.mh_sample_supports(
                X=X,
                y=y_arr01,
                sigma2=sigma2,
                p_feat=0.01,
                beta=beta,
                n_steps=N_STEPS,
                burn_in=BURN_IN,
                target_size=TARGET_NUM_SUPPORT_SETS,
                init_support=starting_support,
                proposal_fnc=proposal_fnc,
            )
        support_sets.append(starting_support)
        mcmc_sampling_and_fitting_time += time() - t0
        # scores.append(MCMCMethod.evaluate_score_laplace(starting_support, X, y, sigma2=sigma2, p_feat=0.01, K=K))

        # get models for each support set
        w_rset = [w_opt]
        for s in tqdm(support_sets, total=len(support_sets)):
            w, _, t0, t1 = fit_ellipsoid_and_sample(s, sample_from_rset)
            if w is None:
                continue
            w_rset.extend(w)
            mcmc_sampling_and_fitting_time += t0
            ellipsoid_sampling_time += t1
        w_rset = np.vstack(w_rset)

        ModelUtils.print_results_summary(
            w_rset, w_opt, 
            bin_X, y, l2, 
            mcmc_sampling_and_fitting_time
        )

        # Create and return result object
        return self.create_result_object(
            method_type=MethodType.MCMC,
            dataset=dn,
            l0=l0,
            l2=l2,
            m=m,
            n_estimators=ne,
            n_support_set=n_support_set,
            n_samples=w_rset.shape[0],
            w_rset=w_rset,
            w_opt=w_opt,
            rset_bound=eps,
            runtime=mcmc_sampling_and_fitting_time,
            ellipsoid_sampling_time=ellipsoid_sampling_time,
            proposal_function=proposal_function,
            sigma2=sigma2,
            beta=beta,
            sample_from_rset=sample_from_rset,
        )

if __name__ == "__main__":
    # Run the ellipsoid method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = MCMCMethod()
    method.run_all_datasets(dataset_settings)