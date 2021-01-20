"""
signatures.py
=======================
Some functions for computing signatures.
"""
# from multiprocessing import cpu_count
from joblib import Parallel, delayed
import numpy as np
import signatory
import torch


def basic_parallel_loop(func, *args, n_jobs):
    """Basic parallel computation loop.

    Args:
        func (function): The function to be applied.
        *args (list): List of arguments [(arg_1_1, ..., arg_n_1), (arg_1, 2), ..., (arg_k_n)]. Each tuple of args is
                      fed into func
        n_jobs (bool): Set False to run a normal for loop (useful when debugging).

    Returns:
        list: Results from the function call.
    """
    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(delayed(func)(*a) for a in args[0])
    else:
        results = []
        for a in args[0]:
            results.append(func(*a))
    return results


def compute_variable_length_signatures(data_list, transform='signature', depth=3, n_jobs=1):
    """Computes signatures over a list of variable lengths by first grouping lengths and batch processing.

    Args:
        data_list (list): A list of variable length time data. Each entry must have shape [L_i, C] with fixed C.
        transform (str): The 'signature' or 'logsignature' transform.
        depth (int): The depth to compute to.
        n_jobs (int): Runs signature computations in parallel.

    Returns:
        tensor: Stacked tensor of signatures.
    """
    def _transform(path):
        return getattr(signatory, transform)(path, depth=depth)

    # First store the lengths of each tensor
    lengths = np.array([s.size(0) for s in data_list])

    # Group indexes and data where the lengths are the same
    index_groups = [np.argwhere(lengths == i).reshape(-1) for i in np.unique(lengths)]
    data_groups = [[torch.stack([data_list[i] for i in idxs])] for idxs in index_groups]

    # Compute the signatures of these same length samples in batch
    signature_groups = basic_parallel_loop(_transform, data_groups, n_jobs=n_jobs)

    # Now stick them back in the right order
    signatures = torch.zeros(len(data_list), signature_groups[0].shape[-1])
    for i, idxs in enumerate(index_groups):
        signatures[idxs] = signature_groups[i]

    return signatures


def compute_group_list_signatures(books, transform='signature', depth=3, n_jobs=1):
    """Computes the signatures of a list containing lists of variable length time series.

    Suppose we have multiple entries each containing multiple variable length time series. This function will compute
    the signatures of each time series and return the signature tensors in the original groups.

    As an example, consider the case where we have multiple books each containing multiple sentences that we model as
    time series'. This function will compute the signatures of each of the sentences and return as a list such that each
    list index corresponds to the sentence signatures of a given book.

    Args:
        books (list): A list of lists of time series of the form described above.
        transform (str): The 'signature' or 'logsignature' transform.
        depth (int): Signature truncation depth.
        n_jobs (int): Parallelisation over the signatures.

    Returns:
        list of Tensors: Each tensor contains the stacked signatures for a given 'book'.
    """
    sentences = [torch.Tensor(s) for b in books for s in b]
    signatures = compute_variable_length_signatures(sentences, depth=depth, transform=transform, n_jobs=n_jobs)
    group_idxs = [0] + list(np.cumsum([len(b) for b in books]))
    group_signatures = [signatures[group_idxs[i]:group_idxs[i+1]] for i in range(len(books))]
    return group_signatures


class LeadLag:
    """ Applies the leadlag transformation to each path.

    Example:
        This is a string man
            [1, 2, 3] -> [[1, 1], [2, 1], [2, 2], [3, 2], [3, 3]]
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Interleave
        X_repeat = X.repeat_interleave(2, dim=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = torch.cat((lead, lag), 2)

        return X_leadlag


