import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle as pkl
import random
import inspect
from functools import partial
from typing import Dict, Any, List, Tuple, Optional
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
                "swap", 
                # "correlation_swap", 
                "multi_swap", 
                "multi_swap_3",
                # "same_feature_swap",
                "feature_swap",
            ],
            "sigma2": [10.0],
            "sample_from_rset": [0, 20],
            "mh_variant": ["standard", "incremental_cd"],
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
    def evaluate_score_laplace_with_state(S, X, y, sigma2, p_feat, K, max_iter=1000):
        """
        Same as evaluate_score_laplace, but also returns (theta_hat, proba)
        so that downstream algorithms can warm-start or update coordinates.
        """
        idx = np.array(sorted(list(S)), dtype=int)
        X_S = X[:, idx]
        theta_hat, proba = MCMCMethod.map_logistic_theta(
            X_S, y, sigma2=sigma2, max_iter=max_iter
        )
        J_S = MCMCMethod.compute_laplace_score_from_state(
            X_S, theta_hat, proba, y, sigma2, p_feat, K
        )
        return J_S, theta_hat, proba

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
    def compute_laplace_score_from_state(X_S, theta, proba, y, sigma2, p_feat, K, eps=1e-12):
        """
        J(S) ≈ log P(S) + log P(D | S, θ) + log P(θ) - 0.5 log det(A_S)
        given design X_S, coefficient vector theta, and probabilities proba.
        """
        log_likelihood = np.sum(
            y * np.log(proba + eps) + (1.0 - y) * np.log(1.0 - proba + eps)
        )
        log_prior_theta = -0.5 * np.dot(theta, theta) / sigma2
        _, logdet = MCMCMethod.hessian_and_logdet(X_S, proba, sigma2=sigma2)
        L_S = log_likelihood + log_prior_theta - 0.5 * logdet
        k = X_S.shape[1]
        log_prior_S = k * np.log(p_feat) + (K - k) * np.log(1.0 - p_feat)
        return L_S + log_prior_S

    @staticmethod
    def evaluate_score_laplace(S, X, y, sigma2, p_feat, K, max_iter=1000):
        """
        J(S) ≈ log P(S) + log P(D | S, θ̂_S) + log P(θ̂_S) - 0.5 log det(A_S)
        """
        idx = np.array(sorted(list(S)), dtype=int)
        X_S = X[:, idx]
        theta_hat, proba = MCMCMethod.map_logistic_theta(X_S, y, sigma2=sigma2, max_iter=max_iter)
        return MCMCMethod.compute_laplace_score_from_state(
            X_S, theta_hat, proba, y, sigma2, p_feat, K
        )

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

    @staticmethod
    def propose_feature_swap(S, K, score, header=None):
        """
        one-step proposal that swaps supports, choosing a new support uniformly from the features
            - choose one support in S to remove
            - choose a new support uniformly from the features
        
        S            : current support (set of indices)
        K            : total number of features (0..K-1)
        header       : header of the dataset
        """
        feature_indices = ModelUtils.get_feature_indices(header)
        del feature_indices['intercept']

        current = set(S)
        in_support = list(current - {INTERCEPT_IDX})
        if len(in_support) == 0:
            return current, 0.0, 0.0
        f_remove = random.choice(in_support)

        # Build support->feature map (for backward probability computation)
        support_to_feature = {}
        for feat, indices in feature_indices.items():
            for idx in indices:
                support_to_feature[idx] = feat

        # ---- Forward proposal probability q(S -> S') ----
        # 1) remove f_remove uniformly from |S|-1
        # 2) pick a feature uniformly from F features
        # 3) pick a support uniformly from that feature's supports, excluding f_remove if present
        features = list(feature_indices.keys())
        F = len(features)
        if F == 0:
            return current, 0.0, 0.0

        add_feat = random.choice(features)
        add_candidates = list(feature_indices[add_feat])
        # Ensure the added support is not already in the current support set.
        # (This also forbids re-adding f_remove since it's in `current` pre-move.)
        add_candidates = [idx for idx in add_candidates if idx not in current]
        if len(add_candidates) == 0:
            return current, 0.0, 0.0
        f_add = random.choice(add_candidates)

        denom_forward = len(add_candidates)
        log_q_forward = -np.log(len(in_support)) - np.log(F) - np.log(denom_forward)

        # ---- Backward proposal probability q(S' -> S) ----
        # In reverse, we'd remove f_add uniformly from |S'|-1 and then re-add f_remove via:
        # pick feature = feature(f_remove) uniformly from F, then pick support f_remove from that feature
        # excluding f_add if it belongs to that same feature.
        remove_feat = support_to_feature.get(f_remove, None)
        if remove_feat is None:
            return current, 0.0, 0.0
        back_candidates = list(feature_indices[remove_feat])
        # Reverse move: starting from S_prop, we first remove f_add, leaving (current \ {f_remove}).
        # So when "adding" in reverse, we must avoid supports already in that set.
        forbidden_back = current - {f_remove}
        back_candidates = [idx for idx in back_candidates if idx not in forbidden_back]
        if f_remove not in back_candidates:
            return current, 0.0, 0.0

        denom_backward = len(back_candidates)
        log_q_backward = -np.log(len(in_support)) - np.log(F) - np.log(denom_backward)
        
        S_prop = set(current)
        S_prop.remove(f_remove)
        S_prop.add(f_add)
        return S_prop, log_q_forward, log_q_backward

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

    @staticmethod
    def mh_sample_supports_incremental_cd(
        X,
        y,
        sigma2=10.0,
        n_steps=2000,
        p_feat=0.01,
        beta=0.025,
        burn_in=500,
        target_size=30,
        init_support=None,
        proposal_fnc=propose_swap,
        full_refit_interval=10,
        cd_steps_on_proposal=10,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Metropolis–Hastings sampler that:
          - uses the usual Laplace score J(S),
          - but avoids re-training LogisticRegression from scratch at every step.

        Full refit is performed when:
          1) it has been `full_refit_interval` steps since the last refit, or
          2) the proposal is not a simple 1-swap, or
          3) the new feature is sufficiently uncorrelated with the current 
             support: refit if max_j |corr(X[:, f_add], X[:, j])| for j in 
             S_current is below that threshold (warm start is then poor).

        At each incremental proposal we warm-start and run
        `cd_steps_on_proposal` CD steps on the proposed support.

        Notes:
          - This variant is designed primarily for simple 1-swap proposals
            such as `propose_swap`. If a proposal modifies more than one
            feature at once, we safely fall back to a full refit.
        """

        def sigmoid(z):
            z = np.clip(z, -20.0, 20.0)
            return 1.0 / (1.0 + np.exp(-z))

        # Bounds to avoid numerical blow-up in CD (match typical L2 logistic scale)
        Z_CLIP = 20.0
        THETA_CLIP = 10.0

        n, K = X.shape

        # ---------- 1. initialize support and MAP state ----------
        if init_support is not None:
            S_current = set(init_support)
            S_current.add(INTERCEPT_IDX)
        else:
            S_current = {INTERCEPT_IDX}

        (
            J_current_raw,
            theta_current,
            proba_current,
        ) = MCMCMethod.evaluate_score_laplace_with_state(
            S_current, X, y, sigma2=sigma2, p_feat=p_feat, K=K
        )

        z_current = np.log(
            np.clip(proba_current, 1e-8, 1.0 - 1e-8)
            / np.clip(1.0 - proba_current, 1e-8, 1.0)
        )

        samples: List[List[int]] = []
        scores: List[float] = []
        accepts = 0
        score = MCMCMethod.compute_corr_score(X, y)

        # number of steps between collected samples
        skip_by = (n_steps - burn_in) // target_size

        # diffs = defaultdict(list)
        # refit_count = 0
        last_refit = 0
        for t in range(n_steps):
            idx_current = np.array(sorted(list(S_current)), dtype=int)

            # Propose new support
            S_prop, log_q_forward, log_q_backward = proposal_fnc(
                S_current, K, score
            )

            # Identify added / removed features (assuming 1-swap most of the time)
            added = list(S_prop - S_current)
            removed = list(S_current - S_prop)

            use_incremental = (
                len(added) == 1 and len(removed) == 1 and full_refit_interval > 0
            )

            # refit when the new feature is sufficiently uncorrelated with current support
            if use_incremental:
                # time0 = time()
                f_add = added[0]
                x_add = X[:, f_add].astype(float)
                x_add_c = x_add - x_add.mean()
                norm_add = np.sqrt((x_add_c ** 2).sum()) + 1e-12
                f_remove = removed[0]
                x_remove = X[:, f_remove].astype(float)
                x_remove_c = x_remove - x_remove.mean()
                norm_remove = np.sqrt((x_remove_c ** 2).sum()) + 1e-12
                abs_corr = np.abs(x_add_c.dot(x_remove_c) / (norm_add * norm_remove))
                
                if abs_corr < 0.01:
                    use_incremental = False
                    # print(
                    #     f"abs_corr: {abs_corr} | "
                    #     f"f_add ({f_add}): {header[f_add]} | "
                    #     f"f_remove ({f_remove}): {header[f_remove]} | "
                    #     f"time: {time() - time0:.6f} seconds"
                    # )

            # Fallback to full refit periodically or when move is not simple
            if last_refit > full_refit_interval or not use_incremental:
                # refit_count += 1
                # time0 = time()
                J_prop_raw = MCMCMethod.evaluate_score_laplace(
                    S_prop, X, y, sigma2=sigma2, p_feat=p_feat, K=K
                )
                theta_prop = None
                proba_prop = None
                z_prop = None
                last_refit = 0
                # print(f"full refit time: {time() - time0:.6f} seconds")
            else:
                # time0 = time()
                # ---------- incremental 1-coordinate update ----------
                f_add = added[0]
                f_remove = removed[0]

                # Map current coefficients by feature index
                feat_to_coef = {
                    f: c for f, c in zip(idx_current, theta_current)
                }
                theta_remove = feat_to_coef.get(f_remove, 0.0)

                # Drop the removed feature from the linear predictor
                z_intermediate = z_current - X[:, f_remove] * theta_remove

                # Probabilities after dropping removed feature
                p_intermediate = sigmoid(z_intermediate)

                # One Newton step for the newly added feature
                x_new = X[:, f_add]
                # Gradient for coordinate j: g_j = X_j^T (p - y) + (1/sigma2) * theta_j
                g_j = x_new.dot(p_intermediate - y)
                # Diagonal Hessian entry: h_jj = X_j^T W X_j + (1/sigma2); clip w to avoid tiny h_jj
                w = np.clip(p_intermediate * (1.0 - p_intermediate), 1e-12, None)
                h_jj = (x_new * x_new * w).sum() + (1.0 / sigma2)

                if h_jj <= 0:
                    theta_new = 0.0
                else:
                    theta_new = np.clip(-g_j / h_jj, -THETA_CLIP, THETA_CLIP)

                # Updated linear predictor and probabilities (clip z to avoid saturation)
                z_prop = np.clip(z_intermediate + x_new * theta_new, -Z_CLIP, Z_CLIP)
                proba_prop = sigmoid(z_prop)

                # Build theta vector aligned with sorted idx_prop
                idx_prop = np.array(sorted(list(S_prop)), dtype=int)
                theta_prop = np.zeros(len(idx_prop))
                for i, f in enumerate(idx_prop):
                    if f == f_add:
                        theta_prop[i] = theta_new
                    elif f in S_current:
                        theta_prop[i] = feat_to_coef.get(f, 0.0)

                X_S_prop = X[:, idx_prop]

                # Run additional CD steps on the full proposed support
                for _ in range(cd_steps_on_proposal - 1):
                    for j in range(len(idx_prop)):
                        x_j = X_S_prop[:, j]
                        g_j = x_j.dot(proba_prop - y) + (1.0 / sigma2) * theta_prop[j]
                        w = np.clip(proba_prop * (1.0 - proba_prop), 1e-12, None)
                        h_jj = (x_j * x_j * w).sum() + (1.0 / sigma2)
                        if h_jj <= 0:
                            continue
                        delta_j = -g_j / h_jj
                        new_theta_j = np.clip(theta_prop[j] + delta_j, -THETA_CLIP, THETA_CLIP)
                        actual_delta = new_theta_j - theta_prop[j]
                        theta_prop[j] = new_theta_j
                        z_prop = np.clip(z_prop + x_j * actual_delta, -Z_CLIP, Z_CLIP)
                        proba_prop = sigmoid(z_prop)

                # Compute Laplace score using the updated (theta_prop, proba_prop)
                J_prop_raw = MCMCMethod.compute_laplace_score_from_state(
                    X_S_prop, theta_prop, proba_prop, y, sigma2, p_feat, K
                )
                last_refit += 1
                # time1 = time()

                # # Debug: compare against full refit score (expensive; prints every step)
                # J_prop_full = MCMCMethod.evaluate_score_laplace(
                #     S_prop, X, y, sigma2=sigma2, p_feat=p_feat, K=K
                # )
                # print(
                #     f"[t={t+1}={(t+1) % full_refit_interval} mod {full_refit_interval}] incremental J={J_prop_raw:.6f} | "
                #     f"full-refit J={J_prop_full:.6f} | "
                #     f"diff={J_prop_raw - J_prop_full:.6f} | "
                #     f"incremental time: {time1 - time0:.6f} seconds"
                # )
                # if J_prop_raw - J_prop_full < -100.0:
                #     print(f"{MAGENTA}YOYO{RESET}")
                # diffs[(t+1) % full_refit_interval].append(J_prop_raw - J_prop_full)

            # MH acceptance ratio (log scale)
            log_alpha = beta * (
                (J_prop_raw - J_current_raw) + (log_q_backward - log_q_forward)
            )

            u = np.random.rand()
            if np.log(u) < min(0.0, log_alpha):
                accepts += 1
                S_current = S_prop
                J_current_raw = J_prop_raw

                # If we computed an incremental state, keep it; otherwise we
                # refresh from scratch for the new support.
                if theta_prop is not None and proba_prop is not None:
                    theta_current = theta_prop
                    proba_current = proba_prop
                    z_current = (
                        z_prop
                        if z_prop is not None
                        else np.log(
                            np.clip(proba_current, 1e-8, 1.0 - 1e-8)
                            / np.clip(1.0 - proba_current, 1e-8, 1.0)
                        )
                    )
                else:
                    (
                        _,
                        theta_current,
                        proba_current,
                    ) = MCMCMethod.evaluate_score_laplace_with_state(
                        S_current,
                        X,
                        y,
                        sigma2=sigma2,
                        p_feat=p_feat,
                        K=K,
                    )
                    z_current = np.log(
                        np.clip(proba_current, 1e-8, 1.0 - 1e-8)
                        / np.clip(1.0 - proba_current, 1e-8, 1.0)
                    )

            # Periodic full refit on the *current* state to keep MAP fresh
            if (t + 1) % full_refit_interval == 0:
                (
                    J_current_raw,
                    theta_current,
                    proba_current,
                ) = MCMCMethod.evaluate_score_laplace_with_state(
                    S_current, X, y, sigma2=sigma2, p_feat=p_feat, K=K
                )
                z_current = np.log(
                    np.clip(proba_current, 1e-8, 1.0 - 1e-8)
                    / np.clip(1.0 - proba_current, 1e-8, 1.0)
                )

            # store after burn-in
            if t >= burn_in and (t + 1) % skip_by == 0:
                samples.append(sorted(list(set(S_current))))
                scores.append(J_current_raw)

            if (t + 1) % skip_by == 0:
                acc_rate = accepts / (t + 1)
                print(
                    f"step {t+1}, |S|={len(S_current)}, J={J_current_raw:.2f}, accept_rate={acc_rate:.3f}"
                )
        
        # # print average diffs
        # for k, v in diffs.items():
        #     print(f"full refit interval {k}: {np.mean(v):.6f}")
        # print(f"refit count: {refit_count}")

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
                      sample_from_rset: int = 0, beta: float = 1.0,
                      mh_variant: str = "standard", full_refit_interval: int = 10, cd_steps_on_proposal: int = 10,
                      **kwargs) -> Any:
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
                sampling="surface", distance_metric=None, r_min=r_min,
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
                "multi_swap_3": {"n_swaps": 3},
                "same_feature_swap": {"header": cum_header},
                "feature_swap": {"header": cum_header},
            }

            # Map proposal function names to base functions
            proposal_map = {
                "swap": MCMCMethod.propose_swap,
                "correlation_swap": MCMCMethod.propose_weighted_swap,
                "multi_swap": MCMCMethod.propose_multi_swap,
                "multi_swap_3": MCMCMethod.propose_multi_swap,
                "same_feature_swap": MCMCMethod.propose_same_feature_swap,
                "feature_swap": MCMCMethod.propose_feature_swap,
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

            if mh_variant == "standard":
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
            elif mh_variant == "incremental_cd":
                support_sets, scores = MCMCMethod.mh_sample_supports_incremental_cd(
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
                    full_refit_interval=full_refit_interval,
                    cd_steps_on_proposal=cd_steps_on_proposal,
                )
            else:
                raise ValueError(
                    f"Unknown mh_variant: {mh_variant}. "
                    f"Expected one of: ['standard', 'incremental_cd']"
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
            mh_variant=mh_variant,
            full_refit_interval=full_refit_interval,
            cd_steps_on_proposal=cd_steps_on_proposal,
        )

if __name__ == "__main__":
    # Run the ellipsoid method on all datasets
    from gam_rs_utils.utils import dataset_settings
    
    method = MCMCMethod()
    method.run_all_datasets(dataset_settings)