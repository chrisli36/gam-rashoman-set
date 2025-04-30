import numpy as np
import sys
# import warnings
# warnings.filterwarnings("ignore")
from fasterrisk.utils import get_support_indices, get_nonsupport_indices, compute_logisticLoss_from_ExpyXB
from fasterrisk.base_model import logRegModel
import math

class State:
    def __init__(self, ExpyXB, beta0, betas, loss):
        self.nonzero_swapped = []
        self.zero_swapped = []
        self.ExpyXB = ExpyXB.copy()
        self.beta0 = beta0.copy()
        self.betas = betas.copy()
        self.loss = loss

def feature_correlation(X, i, j):
    correlation_matrix = np.corrcoef(X[:, i], X[:, j])
    return np.mean(correlation_matrix)

class sparseDiversePoolLogRegModel(logRegModel):
    def __init__(self, X, y, lambda2=1e-8, intercept=True, original_lb=-5, original_ub=5):
        super().__init__(X=X, y=y, lambda2=lambda2, intercept=intercept, original_lb=original_lb, original_ub=original_ub)
        self.total = 0
   
    def getAvailableIndices_for_expansion_but_avoid_l(self, nonsupport, support, l):
        """Get the indices of features that can be added to the support of the current sparse solution

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) The current sparse solution

        Returns
        -------
        available_indices : ndarray
            (1D array with `int` type) The indices of features that can be added to the support of the current sparse solution
        """
        return nonsupport

    def getCorrelation(self, betas_1, betas_2):
        indices_1 = betas_1.nonzero()[0]
        indices_2 = betas_2.nonzero()[0]

        X_subset_1 = self.X[:, indices_1]
        X_subset_2 = self.X[:, indices_2]
        
        correlation_matrix = np.corrcoef(X_subset_1.T, X_subset_2.T)
        return np.mean(correlation_matrix)

    def getDiverseSet(self, beta0, betas, losses, correlation_cutoff):
        # randomize order of the solutions
        inv_losses = 1 / losses
        probabilities = np.array(inv_losses) / np.sum(inv_losses)
        num_losses = len(losses)
        sampled_indices = np.random.choice(num_losses, size=num_losses, replace=False, p=probabilities)

        # greedily select diverse solutions
        diverse_beta0, diverse_betas, diverse_losses = [],[],[]
        for idx in sampled_indices:
            if len(betas) == 0:
                pass
            diverse_beta0.append(beta0[idx])
            diverse_betas.append(betas[idx])
            diverse_losses.append(losses[idx])
        return np.array(diverse_beta0), np.array(diverse_betas), np.array(diverse_losses)

    def getUnique(self, betas, other):
        _, unique_indices = np.unique(betas != 0, axis=0, return_index=True)
        return betas[unique_indices], *self.idx(unique_indices, other)
    
    def idx(self, idx, arr):
        return (a[idx].copy() for a in arr)

    def getSparseDiversePoolBeamSearch(self, gap_tolerance=0.005, beam_size=100, swaps=2, limit_finetuning={"strategy": "no limit"}):
        # get feature set and number of features
        nonzero_indices = get_support_indices(self.betas)
        zero_indices = get_nonsupport_indices(self.betas)
        D = len(nonzero_indices)
        Z = len(zero_indices)
        swaps = min(swaps, Z, D)

        curr_betas = np.expand_dims(self.betas.copy(), axis=0)
        curr_beta0 = np.array([self.beta0])
        curr_ExpyXB = np.array([self.ExpyXB])
        curr_last_ft = np.array([0])

        betas_ss = curr_betas[0, nonzero_indices].dot(curr_betas[0, nonzero_indices])
        global_loss = compute_logisticLoss_from_ExpyXB(curr_ExpyXB[0]) + self.lambda2 * betas_ss

        nonzero_swapped = np.full((1, swaps), None, dtype=object)
        zero_swapped = np.full((1, swaps), None, dtype=object)

        for swap in range(swaps):
            # print(f"swap {swap}")
            total_possibilites = len(curr_betas) * D * Z
            next_betas = np.zeros((total_possibilites, self.p))
            next_beta0 = np.zeros((total_possibilites))
            next_ExpyXB = np.zeros((total_possibilites, self.n))
            next_loss = 1e12 * np.ones((total_possibilites))
            next_last_ft = np.zeros((total_possibilites))

            solutions_found = 0
            for b_idx in range(len(curr_betas)):
                b_start = b_idx * D * Z
                b_end = (1 + b_idx) * D * Z

                next_betas[b_start:b_end] = curr_betas[b_idx].copy()
                next_beta0[b_start:b_end] = curr_beta0[b_idx].copy()
                next_ExpyXB[b_start:b_end] = curr_ExpyXB[b_idx].copy()
                next_last_ft[b_start:b_end] = curr_last_ft[b_idx].copy()

                for old_j_idx, old_j in enumerate(nonzero_indices):
                    if old_j in nonzero_swapped[b_idx] or old_j in zero_swapped[b_idx]:
                        continue

                    bd_start = b_start + old_j_idx * Z
                    d_end = b_start + (1 + old_j_idx) * Z

                    next_betas[bd_start:d_end, old_j] = 0
                    next_ExpyXB[bd_start:d_end] = curr_ExpyXB[b_idx] * np.exp(-self.yXT[old_j] * curr_betas[b_idx, old_j])
                    betas_no_old_j_ss = betas_ss - curr_betas[b_idx, old_j]**2

                    for new_j_idx, new_j in enumerate(zero_indices):
                        bdz_idx = bd_start + new_j_idx
                        if new_j in zero_swapped[b_idx] or new_j in nonzero_swapped[b_idx]:
                            continue

                        for _ in range(10):
                            self.optimize_1step_at_coord(next_ExpyXB[bdz_idx], next_betas[bdz_idx], self.yXT[new_j, :], new_j)

                        betas_new_j_ss = betas_no_old_j_ss + next_betas[bdz_idx, new_j] ** 2
                        loss_bdz = compute_logisticLoss_from_ExpyXB(next_ExpyXB[bdz_idx]) + self.lambda2 * betas_new_j_ss

                        regularized_loss_diff = (loss_bdz - global_loss) / global_loss
                        if regularized_loss_diff < gap_tolerance:
                            do_finetuning = True
                            strategy = limit_finetuning["strategy"]
                            if strategy == "no limit":
                                do_finetuning = True
                            elif strategy  == "no finetuning":
                                do_finetuning = False
                            elif strategy  == "every other":
                                do_finetuning = (swaps == 1 or next_last_ft[bdz_idx] % 2 == 0)
                            elif strategy  == "finetune uncorrelated":
                                do_finetuning = (swaps == 1 or feature_correlation(self.X, old_j, new_j) < limit_finetuning["threshold"])
                            else:
                                raise ValueError(f"Invalid limit_finetuning value: {limit_finetuning}")
                            
                            if do_finetuning:
                                next_ExpyXB[bdz_idx], next_beta0[bdz_idx], next_betas[bdz_idx] = self.finetune_on_current_support(
                                    next_ExpyXB[bdz_idx],
                                    next_beta0[bdz_idx],
                                    next_betas[bdz_idx],
                                )
                                betas_finetuned_new_j_ss = next_betas[bdz_idx].dot(next_betas[bdz_idx])
                                next_loss[bdz_idx] = compute_logisticLoss_from_ExpyXB(next_ExpyXB[bdz_idx]) + self.lambda2 * betas_finetuned_new_j_ss
                            else:
                                next_loss[bdz_idx] = loss_bdz
                            next_last_ft[bdz_idx] += 1
                            solutions_found += 1
                        # elif next_last_ft[bdz_idx] > 0:
                        #     print(next_last_ft[bdz_idx])
                        #     next_ExpyXB[bdz_idx], next_beta0[bdz_idx], next_betas[bdz_idx] = self.finetune_on_current_support(
                        #         next_ExpyXB[bdz_idx],
                        #         next_beta0[bdz_idx],
                        #         next_betas[bdz_idx],
                        #     )
                        #     betas_finetuned_new_j_ss = next_betas[bdz_idx].dot(next_betas[bdz_idx])
                        #     finetuned_loss = compute_logisticLoss_from_ExpyXB(next_ExpyXB[bdz_idx]) + self.lambda2 * betas_finetuned_new_j_ss
                        #     print(loss_bdz - finetuned_loss)

                        #     regularized_loss_diff = (finetuned_loss - global_loss) / global_loss
                        #     if regularized_loss_diff < gap_tolerance:
                        #         next_loss[bdz_idx] = finetuned_loss
                        #         next_last_ft[bdz_idx] = 0
                        #         solutions_found += 1

            # print(f"found {solutions_found} solutions")
            # get the solutions within the gap tolerance
            solution_indices = np.argsort(next_loss)[:solutions_found]

            # of the solutions, keep only unique ones
            _, unique_sub_idx = np.unique(next_betas[solution_indices] != 0, axis=0, return_index=True)
            unique_indices = solution_indices[unique_sub_idx]

            # of the unique solutions, select the top beam_size solutions
            top_b_indices = unique_indices[np.argsort(next_loss[unique_indices])[:beam_size]]
            curr_betas, curr_beta0, curr_ExpyXB, curr_losses, curr_last_ft = self.idx(top_b_indices, [next_betas, next_beta0, next_ExpyXB, next_loss, next_last_ft])

            next_zero_swapped = []
            next_nonzero_swapped = []
            for i, bdz_idx in enumerate(top_b_indices):
                # bdz_idx = b_idx * D * Z + old_j_idx * Z + new_j_idx
                new_j_idx = bdz_idx % Z
                old_j_idx = (bdz_idx // Z) % D
                b_idx = bdz_idx // (D * Z)

                next_zero_swapped.append(zero_swapped[b_idx].copy())
                next_nonzero_swapped.append(nonzero_swapped[b_idx].copy())

                next_zero_swapped[-1][swap] = nonzero_indices[old_j_idx]
                next_nonzero_swapped[-1][swap] = zero_indices[new_j_idx]
            if len(next_nonzero_swapped) == 0:
                return np.empty((0)), np.empty((0, self.p)), np.empty((0))
            zero_swapped = np.vstack(next_zero_swapped)
            nonzero_swapped = np.vstack(next_nonzero_swapped)

            del next_betas
            del next_beta0
            del next_ExpyXB
            del next_loss

        # unscale the coefficients and intercept
        betas = np.zeros((len(curr_beta0), self.p))
        betas[:, self.scaled_feature_indices] = curr_betas[:, self.scaled_feature_indices] / self.X_norm[self.scaled_feature_indices]
        beta0 = curr_beta0 - betas.dot(self.X_mean)

        return beta0, betas, curr_losses

    def getSparseDiversePoolSwapK(self, gap_tolerance=0.005, select_top_m=100, maxAttempts=5, 
                                  swaps=2, fanout_decay=0.6, feature_selection="top", state:State=None):
        curr_betas = state.betas if state else self.betas
        curr_beta0 = state.beta0 if state else self.beta0
        curr_ExpyXB = state.ExpyXB if state else self.ExpyXB

        # get feature set and number of features
        nonzero_indices = get_support_indices(curr_betas)
        zero_indices = get_nonsupport_indices(curr_betas)
        num_support = len(nonzero_indices)
        num_nonsupport = len(zero_indices)
        
        maxAttempts = min(maxAttempts, num_nonsupport)
        total_solutions = 1 + num_support * maxAttempts

        # initialize every candidate solution to start off identical to the original
        # betas
        pool_betas = np.zeros((total_solutions, self.p))
        pool_betas[:, nonzero_indices] = curr_betas[nonzero_indices]
        # beta0
        pool_beta0 = curr_beta0 * np.ones((total_solutions, ))
        # stores the array of exp(y * X * beta) for each candidate
        pool_ExpyXB = np.zeros((total_solutions, self.n))
        pool_ExpyXB[-1] = curr_ExpyXB
        # calculate the square sum of the betas
        betas_squareSum = curr_betas[nonzero_indices].dot(curr_betas[nonzero_indices])
        # stores the loss for each candidate, initialized to a large number
        pool_loss = 1e12 * np.ones((total_solutions, ))
        pool_loss[-1] = compute_logisticLoss_from_ExpyXB(curr_ExpyXB) + self.lambda2 * betas_squareSum

        state = State(curr_ExpyXB, curr_beta0, curr_betas, pool_loss[-1]) if state is None else state
        depth = len(state.nonzero_swapped)
        maxAttempts = math.ceil(maxAttempts * (fanout_decay ** depth))

        totalNum_in_diverseSet = 0
        all_beta0, all_betas, all_losses = [], [], []
        num_nonzero_swaps = math.ceil(num_support * (fanout_decay ** depth))
        num_zero_swaps = min(math.ceil(num_nonsupport * (fanout_decay ** depth)), maxAttempts)

        nonzero_indices = np.random.choice(nonzero_indices, size=num_nonzero_swaps, replace=False)
        for old_j_idx, old_j in enumerate(nonzero_indices):
            pool_start = old_j_idx * maxAttempts
            pool_end = (1 + old_j_idx) * maxAttempts

            # skip if the old_j feature has already been swapped
            if old_j in state.nonzero_swapped or old_j in state.zero_swapped:
                continue
            state.nonzero_swapped.append(old_j)

            # update exp(y * X * beta) and betas to reflect dropping the old_j feature
            pool_ExpyXB[pool_start:pool_end] = curr_ExpyXB * np.exp(-self.yXT[old_j] * curr_betas[old_j])
            pool_betas[pool_start:pool_end, old_j] = 0
            betas_no_old_j_squareSum = betas_squareSum - curr_betas[old_j]**2

            # gets available indices for expansion (any of the zero indices)
            availableIndices = self.getAvailableIndices_for_expansion_but_avoid_l(zero_indices, nonzero_indices, old_j) 
            # calculate the gradient on the available indices
            grad_on_availableIndices = -self.yXT[availableIndices].dot(np.reciprocal(1+pool_ExpyXB[pool_start]))
            abs_grad_on_availableIndices = np.abs(grad_on_availableIndices)

            # determine swapping order of features based on their absolute gradients
            if feature_selection == "top":
                new_js = availableIndices[np.argsort(-abs_grad_on_availableIndices)[:maxAttempts]]
            elif feature_selection == "bottom":
                new_js = availableIndices[np.argsort(abs_grad_on_availableIndices)[:maxAttempts]]
            elif feature_selection == "random":
                epsilon = 1e-6
                weights = (abs_grad_on_availableIndices + epsilon)  / np.sum(abs_grad_on_availableIndices + epsilon)
                new_js = np.random.choice(availableIndices, size=num_zero_swaps, replace=False, p=weights)

            for new_j_idx, new_j in enumerate(new_js):
                pool_idx = pool_start + new_j_idx

                # skip if the new_j feature has already been swapped
                if new_j in state.zero_swapped or new_j in state.nonzero_swapped:
                    continue
                state.zero_swapped.append(new_j)

                # perform coordinate descent on the new feature
                for _ in range(10):
                    self.optimize_1step_at_coord(pool_ExpyXB[pool_idx], pool_betas[pool_idx], self.yXT[new_j, :], new_j)
                
                # calculate the loss for the new feature
                betas_new_j_squareSum = betas_no_old_j_squareSum + pool_betas[pool_idx, new_j] ** 2
                loss_sparseDiversePool_index = compute_logisticLoss_from_ExpyXB(pool_ExpyXB[pool_idx]) + self.lambda2 * betas_new_j_squareSum

                # check if the loss is within the gap tolerance
                regularized_loss_diff = (loss_sparseDiversePool_index - state.loss) / state.loss
                if regularized_loss_diff < gap_tolerance:
                    totalNum_in_diverseSet += 1
                    # further finetune the solution
                    pool_ExpyXB[pool_idx], pool_beta0[pool_idx], pool_betas[pool_idx] = self.finetune_on_current_support(
                        pool_ExpyXB[pool_idx],
                        pool_beta0[pool_idx],
                        pool_betas[pool_idx],
                    )
                    # record the loss for the new feature
                    betas_finetuned_new_j_squareSum = pool_betas[pool_idx].dot(pool_betas[pool_idx])
                    pool_loss[pool_idx] = compute_logisticLoss_from_ExpyXB(pool_ExpyXB[pool_idx]) + self.lambda2 * betas_finetuned_new_j_squareSum
                    
                    # update state
                    state.ExpyXB = pool_ExpyXB[pool_idx].copy()
                    state.beta0 = pool_beta0[pool_idx].copy()
                    state.betas = pool_betas[pool_idx].copy()
                    if swaps > 1:
                        top_m_beta0, top_m_betas, top_m_losses = self.getSparseDiversePoolSwapK(
                            gap_tolerance = gap_tolerance,
                            select_top_m = select_top_m,
                            maxAttempts = maxAttempts,
                            swaps = swaps - 1,
                            fanout_decay=fanout_decay,
                            feature_selection=feature_selection,
                            state = state,
                        )
                        all_beta0.append(top_m_beta0.copy())
                        all_betas.append(top_m_betas.copy())
                        all_losses.append(top_m_losses.copy())

                # allow new_j feature to be swapped
                state.zero_swapped.pop()
            # allow old_j feature to be swapped
            state.nonzero_swapped.pop()

        if swaps > 1:
            if len(all_beta0) == 0:
                return np.empty((0)), np.empty((0, self.p)), np.empty((0))
            
            # concatenate all solutions
            all_beta0 = np.hstack(all_beta0)
            all_betas = np.vstack(all_betas)
            all_losses = np.hstack(all_losses)

            if len(all_beta0) == 0:
                return np.empty((0)), np.empty((0, self.p)), np.empty((0))

            # remove duplicate solutions
            _, unique_indices = np.unique(all_betas != 0, axis=0, return_index=True)
            diverse_beta0 = all_beta0[unique_indices]
            diverse_betas = all_betas[unique_indices, :]
            diverse_losses = all_losses[unique_indices]

            # # greedily select diverse solutions
            # diverse_beta0, diverse_betas, diverse_losses = self.getDiverseSet(diverse_beta0, diverse_betas, diverse_losses, correlation_cutoff)

            # take the top m best solutions
            if select_top_m == -1:
                top_m = np.argsort(diverse_losses)
            else:
                top_m = np.argsort(diverse_losses)[:select_top_m]
            return diverse_beta0[top_m], diverse_betas[top_m], diverse_losses[top_m]

        # select top m solutions
        if select_top_m == -1:
            top_m = np.argsort(pool_loss[:total_solutions - 1])[:totalNum_in_diverseSet]
        else:
            top_m = np.argsort(pool_loss[:total_solutions - 1])[:totalNum_in_diverseSet][:select_top_m]

        # unscale the coefficients and intercept
        top_m_original_betas = np.zeros((len(top_m), self.p))
        top_m_original_betas[:, self.scaled_feature_indices] = pool_betas[top_m][:, self.scaled_feature_indices] / self.X_norm[self.scaled_feature_indices]
        top_m_original_beta0 = pool_beta0[top_m] - top_m_original_betas.dot(self.X_mean)

        return top_m_original_beta0, top_m_original_betas, pool_loss[top_m]

    def get_sparseDiversePool(self, gap_tolerance=0.05, select_top_m=10, maxAttempts=50):
        """For the current sparse solution, get from the sparse diverse pool [select_top_m] solutions, which perform equally well as the current sparse solution. This sparse diverse pool is also called the Rashomon set. We discover new solutions by swapping 1 feature in the support of the current sparse solution.

        Parameters
        ----------
        gap_tolerance : float, optional
            New solution is accepted after swapping features if the new loss is within the [gap_tolerance] of the old loss, by default 0.05
        select_top_m : int, optional
            We select the top [select_top_m] solutions from support_size*maxAttempts number of new solutions, by default 10
        maxAttempts : int, optional
            We try to swap each feature in the support with [maxAttempts] of new features, by default 50

        Returns
        -------
        intercept_array : ndarray
            (1D array with `float` type) Return the intercept array with shape = (select_top_m, )
        coefficients_array : ndarray
            (2D array with `float` type) Return the coefficients array with shape = (select_top_m, p)
        """
        # select top m solutions with the lowest logistic losses
        # Note Bene: loss comparison here does not include logistic loss
        nonzero_indices = get_support_indices(self.betas)
        zero_indices = get_nonsupport_indices(self.betas)

        num_support = len(nonzero_indices)
        num_nonsupport = len(zero_indices)

        maxAttempts = min(maxAttempts, num_nonsupport)
        max_num_new_js = maxAttempts

        total_solutions = 1 + num_support * maxAttempts
        sparseDiversePool_betas = np.zeros((total_solutions, self.p))
        sparseDiversePool_betas[:, nonzero_indices] = self.betas[nonzero_indices]

        sparseDiversePool_beta0 = self.beta0 * np.ones((total_solutions, ))
        sparseDiversePool_ExpyXB = np.zeros((total_solutions, self.n))
        sparseDiversePool_loss = 1e12 * np.ones((total_solutions, ))

        sparseDiversePool_ExpyXB[-1] = self.ExpyXB
        sparseDiversePool_loss[-1] = compute_logisticLoss_from_ExpyXB(self.ExpyXB) + self.lambda2 * self.betas[nonzero_indices].dot(self.betas[nonzero_indices])

        betas_squareSum = self.betas[nonzero_indices].dot(self.betas[nonzero_indices])

        totalNum_in_diverseSet = 1
        for num_old_j, old_j in enumerate(nonzero_indices):
            # pick $maxAttempt$ number of features that can replace old_j
            sparseDiversePool_start = num_old_j * maxAttempts
            sparseDiversePool_end = (1 + num_old_j) * maxAttempts

            sparseDiversePool_ExpyXB[sparseDiversePool_start:sparseDiversePool_end] = self.ExpyXB * np.exp(-self.yXT[old_j] * self.betas[old_j])

            sparseDiversePool_betas[sparseDiversePool_start:sparseDiversePool_end, old_j] = 0
            
            betas_no_old_j_squareSum = betas_squareSum - self.betas[old_j]**2

            availableIndices = self.getAvailableIndices_for_expansion_but_avoid_l(zero_indices, nonzero_indices, old_j) 

            grad_on_availableIndices = -self.yXT[availableIndices].dot(np.reciprocal(1+sparseDiversePool_ExpyXB[sparseDiversePool_start]))
            abs_grad_on_availableIndices = np.abs(grad_on_availableIndices)

            # new_js = np.argpartition(abs_full_grad, -max_num_new_js)[-max_num_new_js:]
            new_js = availableIndices[np.argsort(-abs_grad_on_availableIndices)[:max_num_new_js]]

            for num_new_j, new_j in enumerate(new_js):
                sparseDiversePool_index = sparseDiversePool_start + num_new_j
                for _ in range(10):
                    self.optimize_1step_at_coord(sparseDiversePool_ExpyXB[sparseDiversePool_index], sparseDiversePool_betas[sparseDiversePool_index], self.yXT[new_j, :], new_j)
                
                loss_sparseDiversePool_index = compute_logisticLoss_from_ExpyXB(sparseDiversePool_ExpyXB[sparseDiversePool_index]) + self.lambda2 * (betas_no_old_j_squareSum + sparseDiversePool_betas[sparseDiversePool_index, new_j] ** 2)

                if (loss_sparseDiversePool_index - sparseDiversePool_loss[-1]) / sparseDiversePool_loss[-1] < gap_tolerance:
                    totalNum_in_diverseSet += 1

                    sparseDiversePool_ExpyXB[sparseDiversePool_index], sparseDiversePool_beta0[sparseDiversePool_index], sparseDiversePool_betas[sparseDiversePool_index] = self.finetune_on_current_support(sparseDiversePool_ExpyXB[sparseDiversePool_index], sparseDiversePool_beta0[sparseDiversePool_index], sparseDiversePool_betas[sparseDiversePool_index])

                    sparseDiversePool_loss[sparseDiversePool_index] = compute_logisticLoss_from_ExpyXB(sparseDiversePool_ExpyXB[sparseDiversePool_index]) + self.lambda2 * (betas_no_old_j_squareSum + sparseDiversePool_betas[sparseDiversePool_index, new_j] ** 2)

        selected_sparseDiversePool_indices = np.argsort(sparseDiversePool_loss)[:totalNum_in_diverseSet][:select_top_m]

        top_m_original_betas = np.zeros((len(selected_sparseDiversePool_indices), self.p))
        top_m_original_betas[:, self.scaled_feature_indices] = sparseDiversePool_betas[selected_sparseDiversePool_indices][:, self.scaled_feature_indices] / self.X_norm[self.scaled_feature_indices]
        top_m_original_beta0 = sparseDiversePool_beta0[selected_sparseDiversePool_indices] - top_m_original_betas.dot(self.X_mean)

        return top_m_original_beta0, top_m_original_betas

        original_sparseDiversePool_solution[1:] = sparseDiversePool_betas[selected_sparseDiversePool_indices].T
        original_sparseDiversePool_solution[1+self.scaled_feature_indices] /= self.X_norm[self.scaled_feature_indices].reshape(-1, 1)

        original_sparseDiversePool_solution[0] = sparseDiversePool_beta0[selected_sparseDiversePool_indices]
        original_sparseDiversePool_solution[0] -= self.X_mean.T @ original_sparseDiversePool_solution[1:]
        
        return original_sparseDiversePool_solution # (1+p, m) m is the number of solutions in the pool

class groupSparseDiversePoolLogRegModel(sparseDiversePoolLogRegModel):
    def __init__(self, X, y, lambda2=1e-8, intercept=True, original_lb=-5, original_ub=5, group_sparsity=10, featureIndex_to_groupIndex=None, groupIndex_to_featureIndices=None):
        super().__init__(X=X, y=y, lambda2=lambda2, intercept=intercept, original_lb=original_lb, original_ub=original_ub)

        self.group_sparsity = group_sparsity
        self.featureIndex_to_groupIndex = featureIndex_to_groupIndex
        self.groupIndex_to_featureIndices = groupIndex_to_featureIndices
    
    def getAvailableIndices_for_expansion_but_avoid_l(self, nonsupport, support, l):
        """Get the indices of features that can be added to the support of the current sparse solution

        Parameters
        ----------
        nonsupport : ndarray
            (1D array with `int` type) The indices of features that are not in the support of the current sparse solution
        support : ndarray
            (1D array with `int` type) The indices of features that are in the support of the current sparse solution
        l : int
            The index of the feature that is to be removed from the support of the current sparse solution and this index l belongs to support

        Returns
        -------
        available_indices : ndarray
            (1D array with `int` type) The indices of features that can be added to the support of the current sparse solution when we delete index l
        """
        existing_groupIndices, freq_existing_groupIndices = np.unique(self.featureIndex_to_groupIndex[support], return_counts=True)
        freq_groupIndex_of_l = freq_existing_groupIndices[existing_groupIndices == self.featureIndex_to_groupIndex[l]]
        if len(existing_groupIndices) < self.group_sparsity:
            # we have not reached the group size yet 
            available_indices = nonsupport
        elif freq_groupIndex_of_l == 1:
            # or if we remove index l, we still do not reach the group size
            available_indices = nonsupport
        else:
            # we reach the group size even if we remove index l
            available_indices = set()
            for groupIndex in existing_groupIndices:
                available_indices.update(self.groupIndex_to_featureIndices[groupIndex])
            available_indices = available_indices - set(support)
            available_indices = np.array(list(available_indices), dtype=int)

        return available_indices
