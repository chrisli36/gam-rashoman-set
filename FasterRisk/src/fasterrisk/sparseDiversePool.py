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

    def getSparseDiversePoolSwapK(self, gap_tolerance=0.005, select_top_m=100, maxAttempts=5, swaps=2, correlation_cutoff=0.5, fanout_decay=0.6, state:State=None):
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
        # maxAttempts = math.ceil(maxAttempts * (fanout_decay ** depth))

        totalNum_in_diverseSet = 0
        all_beta0_solutions, all_betas_solutions, all_pool_losses = [], [], []
        # num_nonzero_swaps = math.ceil(len(nonzero_indices) * (fanout_decay ** depth))
        # nonzero_indices = np.random.choice(nonzero_indices, size=num_nonzero_swaps, replace=False)
        for num_old_j, old_j in enumerate(nonzero_indices):
            if depth == 0:
                print(num_old_j)
            pool_start = num_old_j * maxAttempts
            pool_end = (1 + num_old_j) * maxAttempts

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
            # pick top features largest absolute gradient to swap
            new_js = availableIndices[np.argsort(-abs_grad_on_availableIndices)[:maxAttempts]]

            # epsilon = 1e-6
            # weights = (abs_grad_on_availableIndices + epsilon)  / np.sum(abs_grad_on_availableIndices + epsilon)
            # num_zero_swaps = math.ceil(len(availableIndices) * (fanout_decay ** depth))
            # new_js = np.random.choice(availableIndices, size=num_zero_swaps, replace=False, p=weights)
            for num_new_j, new_j in enumerate(new_js):
                pool_idx = pool_start + num_new_j

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
                # regularized_loss_diff = (loss_sparseDiversePool_index - pool_loss[-1]) / pool_loss[-1]
                if regularized_loss_diff < gap_tolerance:
                    # print(f"swap {swaps}, loss diff: {regularized_loss_diff}")
                    totalNum_in_diverseSet += 1
                    # further finetune the solution
                    pool_ExpyXB[pool_idx], pool_beta0[pool_idx], pool_betas[pool_idx] = self.finetune_on_current_support(
                        pool_ExpyXB[pool_idx], 
                        pool_beta0[pool_idx], 
                        pool_betas[pool_idx]
                    )
                    # record the loss for the new feature
                    # betas_finetuned_new_j_squareSum = betas_no_old_j_squareSum + pool_betas[pool_idx, new_j] ** 2
                    betas_finetuned_new_j_squareSum = pool_betas[pool_idx].dot(pool_betas[pool_idx])
                    pool_loss[pool_idx] = compute_logisticLoss_from_ExpyXB(pool_ExpyXB[pool_idx]) + self.lambda2 * betas_finetuned_new_j_squareSum
                    
                    # update state
                    state.ExpyXB = pool_ExpyXB[pool_idx].copy()
                    state.beta0 = pool_beta0[pool_idx].copy()
                    state.betas = pool_betas[pool_idx].copy()
                    # if swaps > 1 and regularized_loss_diff / gap_tolerance < 0.8:
                    if swaps > 1:
                        # print(f"swaps is {swaps}: {regularized_loss_diff}")
                        # state.loss_tracker.append((loss_sparseDiversePool_index - pool_loss[-1]) / pool_loss[-1])
                        top_m_beta0, top_m_betas, top_m_losses = self.getSparseDiversePoolSwapK(
                            gap_tolerance = gap_tolerance, 
                            select_top_m = select_top_m, 
                            maxAttempts = maxAttempts, 
                            swaps = swaps - 1, 
                            state = state
                        )
                        all_beta0_solutions.append(top_m_beta0.copy())
                        all_betas_solutions.append(top_m_betas.copy())
                        all_pool_losses.append(top_m_losses.copy())
                        # print(f"swaps is {swaps}:, {top_m_losses}")

                # allow new_j feature to be swapped
                state.zero_swapped.pop()
            # allow old_j feature to be swapped
            state.nonzero_swapped.pop()

        if swaps > 1:
            if len(all_beta0_solutions) == 0:
                return np.empty((0)), np.empty((0, self.p)), np.empty((0))
            
            # concatenate all solutions
            all_beta0_solutions = np.hstack(all_beta0_solutions)
            all_betas_solutions = np.vstack(all_betas_solutions)
            all_pool_losses = np.hstack(all_pool_losses)

            if len(all_beta0_solutions) == 0:
                return np.empty((0)), np.empty((0, self.p)), np.empty((0))

            # remove duplicate solutions
            mask = (all_betas_solutions != 0)
            _, unique_indices = np.unique(mask, axis=0, return_index=True)
            unique_beta0 = all_beta0_solutions[unique_indices]
            unique_betas = all_betas_solutions[unique_indices, :]
            unique_losses = all_pool_losses[unique_indices]

            diverse_beta0 = unique_beta0
            diverse_betas = unique_betas
            diverse_losses = unique_losses

            # randomize order of the solutions
            inv_losses = 1 / unique_losses
            probabilities = np.array(inv_losses) / np.sum(inv_losses)
            num_losses = len(unique_losses)
            sampled_indices = np.random.choice(num_losses, size=num_losses, replace=False, p=probabilities)

            # greedily select diverse solutions
            diverse_betas = [unique_betas[sampled_indices[0]]]
            diverse_beta0 = [unique_beta0[sampled_indices[0]]]
            diverse_losses = [unique_losses[sampled_indices[0]]]
            for i, betas in enumerate(all_betas_solutions[sampled_indices[1:]]):
                max_correlation = max([self.getCorrelation(betas, db) for db in diverse_betas])
                print("hi:", max_correlation)
                if max_correlation < correlation_cutoff:
                    diverse_betas.append(betas)
                    diverse_beta0.append(unique_beta0[sampled_indices[i]])
                    diverse_losses.append(unique_losses[sampled_indices[i]])
            diverse_betas = np.array(diverse_betas)
            diverse_beta0 = np.array(diverse_beta0)
            diverse_losses = np.array(diverse_losses)
            print(f"there were {len(unique_betas)} solutions, {len(diverse_betas)} of them were diverse")

            # take the top m best solutions
            top_m_indices = np.argsort(diverse_losses)[:len(diverse_losses)][:select_top_m]
            top_m_betas = diverse_betas[top_m_indices]
            top_m_beta0 = diverse_beta0[top_m_indices]
            top_m_losses = diverse_losses[top_m_indices]

            return top_m_beta0, top_m_betas, top_m_losses

        # select top m solutions
        selected_indices = np.argsort(pool_loss[:total_solutions - 1])[:totalNum_in_diverseSet][:select_top_m]
        top_m_pool_losses = pool_loss[selected_indices]
        # if len(selected_indices) != 0:
        #     print(f"swaps is 1: {state.nonzero_swapped}, {state.zero_swapped}, {top_m_pool_losses}, {selected_indices}")

        # unscale the coefficients and intercept
        top_m_original_betas = np.zeros((len(selected_indices), self.p))
        top_m_original_betas[:, self.scaled_feature_indices] = pool_betas[selected_indices][:, self.scaled_feature_indices] / self.X_norm[self.scaled_feature_indices]
        top_m_original_beta0 = pool_beta0[selected_indices] - top_m_original_betas.dot(self.X_mean)

        # # count the number of solutions whose row mask matches [ 6  7  8 12 13 14 18 23 29 37 ]
        # for row in top_m_original_betas:
        #     if np.array_equal(row.nonzero()[0], np.array([ 6,  7,  8, 12, 13, 14, 18, 23, 29, 37 ])):
        #         print("hiiiii", state.nonzero_swapped, state.zero_swapped)
        #         self.total += 1

        return top_m_original_beta0, top_m_original_betas, top_m_pool_losses

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
