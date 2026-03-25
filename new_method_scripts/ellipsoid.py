"""
Ellipsoid Sampling Baseline for GAM Rashomon Sets

Samples models from the Rashomon set by:
  1. Fitting MAP logistic regression on a given support
  2. Computing the Hessian to define an ellipsoid around the MAP
  3. Sampling weight vectors uniformly from the ellipsoid
  4. Filtering by log posterior (NLL) bound: R(eps) = {w : nll(w) <= best_nll * (1 + eps)}

The ellipsoid approximates the Rashomon set via the quadratic Taylor
expansion of the negative log-posterior around the MAP estimate:

    0.5 * (w - w*)^T H (w - w*) <= ub

When best_nll is provided (recommended, aligned with mcmc_rashomon):
    ub = nll_center - best_nll  (slack in log-posterior units)

Otherwise (legacy): ub = 2 * eps * loss(w*) * scale

Usage:
    from new_method_scripts.ellipsoid import EllipsoidSampler

    sampler = EllipsoidSampler(X, y, eps=0.05)
    result = sampler.sample(n_samples=10000)
"""

import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from typing import Optional, List
from time import time
from new_method_scripts.mcmc_rashomon import RashomonResult


class EllipsoidSampler:
    """Ellipsoid sampling baseline for the log-posterior Rashomon set."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 0.05,
        sigma2: float = 10.0,
    ):
        """
        Args:
            X: (n, p) design matrix (binarized, with intercept column).
            y: (n,) binary labels in {0, 1}.
            eps: Rashomon set tolerance.
            sigma2: Prior variance on weights (= C in sklearn).
        """
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.n, self.p = self.X.shape
        self.eps = eps
        self.sigma2 = sigma2

    # ================================================================== #
    #  Fitting
    # ================================================================== #

    @staticmethod
    def fit_logistic(X_S, y, sigma2=10.0, max_iter=1000):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l2", C=sigma2, fit_intercept=False,
                solver="lbfgs", max_iter=max_iter,
            )
            clf.fit(X_S, y)
        return clf.coef_.ravel(), clf.predict_proba(X_S)[:, 1]

    @staticmethod
    def compute_hessian(X_S, proba, sigma2):
        """Hessian of the negative log-posterior at the MAP estimate."""
        p = np.clip(proba, 1e-8, 1 - 1e-8)
        W = p * (1 - p)
        return (X_S.T * W) @ X_S + np.eye(X_S.shape[1]) / sigma2

    @staticmethod
    def total_nll(proba, y, theta, sigma2):
        """Total negative log-posterior (NLL + L2 prior)."""
        _e = 1e-12
        nll = -np.sum(y * np.log(proba + _e) + (1 - y) * np.log(1 - proba + _e))
        return nll + 0.5 * np.dot(theta, theta) / sigma2

    @staticmethod
    def misclassification_error(proba, y):
        return float(np.mean((proba >= 0.5).astype(float) != y))

    # ================================================================== #
    #  Ellipsoid sampling
    # ================================================================== #

    @staticmethod
    def _sample_ellipsoid(theta_center, eigvals, eigvecs, ub,
                          n_samples, sampling="uniform"):
        """
        Draw from  {w : 0.5 (w-c)^T H (w-c) <= ub}  where H = V diag(S) V^T.
        Semi-axes are sqrt(2*ub / eigenvalue_i).
        """
        d = len(theta_center)
        a = np.sqrt(2 * ub / np.maximum(eigvals, 1e-12))

        u = np.random.randn(n_samples, d)
        u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
        r = np.ones(n_samples) if sampling != "uniform" else np.random.random(n_samples) ** (1.0 / d)
        z = u * r[:, None]  # uniform in unit ball (or surface if sampling != "uniform")

        # Rotate then scale using eigvecs as columns
        deltas = (eigvecs * a) @ z.T  # shape (d, n_samples)
        theta_samples = theta_center[:, None] + deltas  # (d, n_samples)
        theta_samples = theta_samples.T  # (n_samples, d)

        return theta_samples

    # ================================================================== #
    #  Initial support
    # ================================================================== #

    def find_initial_support(self, l1_C=1.0):
        """Sparse starting support via L1-regularized logistic regression."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                penalty="l1", C=l1_C, solver="saga",
                fit_intercept=False, max_iter=5000,
            )
            clf.fit(self.X, self.y)
        support = np.nonzero(clf.coef_.ravel())[0].tolist()
        if not support:
            corr = np.abs(self.X.T @ (self.y - self.y.mean()))
            support = np.argsort(corr)[-5:].tolist()
        return support

    # ================================================================== #
    #  Main interface
    # ================================================================== #

    def sample(
        self,
        n_samples: int = 10_000,
        support: Optional[List[int]] = None,
        w_center: Optional[np.ndarray] = None,
        sampling: str = "uniform",
        eps: Optional[float] = None,
        l1_C: float = 1.0,
        scale: float = 1.0,
        rashomon_bound: Optional[float] = None,
        best_nll: Optional[float] = None,
    ) -> RashomonResult:
        """
        Sample models from R(eps) = {w : nll(w) <= best_nll * (1 + eps)}.

        Args:
            n_samples:  Candidate weight vectors to draw from the ellipsoid.
            support:    Feature indices to use (L1-selected if None).
            w_center:   Full (p,) weight vector to center the ellipsoid on.
                        If provided, ``support`` is inferred from its nonzero
                        entries (unless ``support`` is also given).  This lets
                        you reuse the MAP estimate from another method (e.g.
                        ``mcmc_result.w_opt``).
            sampling:   "uniform" (interior) or "surface".
            eps:        Override self.eps.
            l1_C:       L1 strength for automatic support selection.
            scale:      Multiplier on the ellipsoid radius (>1 widens the
                        ellipsoid for better coverage at lower acceptance).
                        Only used when best_nll is None.
            rashomon_bound: Deprecated (error-based). Use eps for NLL-based filtering.
            best_nll:   Reference NLL (min over MAPs). When provided, ellipsoid
                        bound is ub = nll_center - best_nll (log-likelihood based,
                        same as mcmc_rashomon). When None, uses eps-based bound.
        Returns:
            RashomonResult
        """
        eps = eps if eps is not None else self.eps
        t0 = time()

        # 1. Support & center
        if w_center is not None and support is None:
            support = np.nonzero(w_center)[0].tolist()
        if support is None:
            support = self.find_initial_support(l1_C)
        idx = sorted(support)
        X_S = self.X[:, idx]
        d = len(idx)

        # 2. MAP estimate — use provided center or fit from scratch
        if w_center is not None:
            theta = w_center[idx]
            proba = 1.0 / (1.0 + np.exp(-np.clip(X_S @ theta, -20, 20)))
        else:
            theta, proba = self.fit_logistic(X_S, self.y, self.sigma2)
        best_err = self.misclassification_error(proba, self.y)
        nll_center = self.total_nll(proba, self.y, theta, self.sigma2)
        w_opt = np.zeros(self.p)
        w_opt[idx] = theta

        # 3. Hessian & ellipsoid bound (aligned with mcmc_rashomon: log-likelihood based)
        H = self.compute_hessian(X_S, proba, self.sigma2)
        eigvals, eigvecs = np.linalg.eigh(H)
        if best_nll is not None:
            # Same as mcmc_rashomon: ub = nll_center - best_nll (slack in log-posterior)
            ub = max(nll_center - best_nll, 1e-6) * scale
        else:
            # Legacy: eps-based bound
            eps_eff = (rashomon_bound - best_err) if rashomon_bound is not None else eps
            eps_eff = max(eps_eff, 1e-8)  # avoid negative ub
            loss_opt = nll_center
            ub = 2 * eps_eff * loss_opt * scale

        print(f"Support: {d} features, MAP error: {best_err:.4f}")
        print(f"Ellipsoid ub={ub:.2f}, sampling {n_samples} candidates...")

        # 4. Sample from ellipsoid
        theta_samples = self._sample_ellipsoid(
            theta, eigvals, eigvecs, ub, n_samples, sampling
        )

        # 5. Vectorized filtering by log posterior (Rashomon bound: nll <= best_nll * (1 + eps))
        logits = X_S @ theta_samples.T                     # (n, n_samples)
        proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        preds = (logits >= 0).astype(np.float64)           # (n, n_samples)
        errors = np.mean(preds != self.y[:, None], axis=0)  # (n_samples,)

        _e = 1e-12
        nlls = np.array([
            -np.sum(self.y * np.log(proba[:, i] + _e) + (1 - self.y) * np.log(1 - proba[:, i] + _e))
            + 0.5 * np.dot(theta_samples[i], theta_samples[i]) / self.sigma2
            for i in range(theta_samples.shape[0])
        ])
        best_nll_ref = nll_center if best_nll is None else min(nll_center, best_nll)
        nll_bound = best_nll_ref * (1 + eps)
        mask = nlls <= nll_bound

        bound = float(np.max(errors[mask])) if mask.any() else best_err
        eps_out = eps

        w_models = np.zeros((int(mask.sum()), self.p))
        w_models[:, idx] = theta_samples[mask]
        kept_errors = errors[mask]

        rt = time() - t0
        precision = mask.sum() / max(n_samples, 1)
        print(f"Best error: {best_err:.4f}, bound: {bound:.4f}")
        print(
            f"Rashomon set: {mask.sum()}/{n_samples} "
            f"({precision:.1%} acceptance), runtime: {rt:.1f}s"
        )

        return RashomonResult(
            w_models=w_models,
            w_opt=w_opt,
            support_sets=[idx] * int(mask.sum()),
            errors=kept_errors,
            best_error=best_err,
            error_bound=bound,
            eps=eps_out,
            n_explored=n_samples,
            n_accepted=int(mask.sum()),
            runtime=rt,
            acceptance_rate=float(precision),
        )
