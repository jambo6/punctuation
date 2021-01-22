import os
import numpy as np
import torch
import signatory
from signatures.compute import compute_variable_length_signatures, basic_parallel_loop
from tqdm import tqdm


def convert_2d_numpy_list_to_3d_tensor(numpy_list):
    return [torch.tensor(x).unsqueeze(0) for x in numpy_list]


def cumulative_sum_over_list_of_tensors(tensor_list, axis=0):
    return [t.cumsum(axis) for t in tensor_list]


def concat_list_tensors(tensor_list_1, tensor_list_2):
    return [torch.cat((t1, t2), axis=1) for t1, t2 in zip(tensor_list_1, tensor_list_2)]


def compute_and_save_signatures_of_books(fnames):
    # Load data
    directory = '../data/processed/Periodic/Punctuation'
    npz_files = [np.load('{}/{}'.format(directory, fname), allow_pickle=True) for fname in fnames]

    # Get word and punctuation sequences
    tupled_data = [
        (npz['words_sequence'], npz['punctuation_sequence'], npz['binary_punctuation_sequence'])
        for npz in npz_files
    ]

    # Now compute the signatures of:
    #   1. Normal and binary punctuation only
    #   2. Normal and binary with time
    #   3. Normal and binary with words
    #   3. Normal and binary with words and time
    # First extract
    normal_punctuation = [torch.tensor(t[1].cumsum(0), dtype=float) for t in tupled_data]
    binary_punctuation = [torch.tensor(t[2].cumsum(0), dtype=float) for t in tupled_data]
    words = [torch.tensor(t[0].cumsum(0).reshape(-1, 1), dtype=float) for t in tupled_data]
    times = [torch.arange(0, len(t)).reshape(-1, 1).to(float) for t in words]
    # Compute the signatures
    signature_dict = {}
    channels_dict = {}
    for type in ['normal', 'binary']:
        for use_words in [True, False]:
            for use_times in [True, False]:
                times_str, words_str, punctuation_str = '', '', '{}'.format(type)
                punctuation = normal_punctuation if type == 'normal' else binary_punctuation
                depth = 4 if type == 'normal' else 6
                if use_words:
                    words_str = '_words'
                    punctuation = concat_list_tensors(words, punctuation)
                if use_times:
                    times_str = '_times'
                    punctuation = concat_list_tensors(times, punctuation)
                signatures = compute_variable_length_signatures(punctuation, transform='signature', depth=depth)
                string = punctuation_str + words_str + times_str
                # Get signature depths
                channels = punctuation[0].shape[-1]
                signature_channels = [signatory.signature_channels(channels, x) for x in range(1, depth + 1)]
                # For saving
                signature_dict[string] = signatures.numpy()
                channels_dict[string + '_channels'] = signature_channels

    # Now also
    for i, index in enumerate(fnames):
        save_dict = {key: value[i] for key, value in signature_dict.items()}
        np.savez(
            directory + '/signatures/{}'.format(fnames[i]),
            **save_dict
        )

    # Save information about the channels
    np.savez(
        directory + '/signatures/channel_information.npz',
        **channels_dict
    )


if __name__ == '__main__':
    directory = '../data/processed/Periodic/Punctuation'
    filenames = os.listdir(directory)
    filenames = [f for f in filenames if f.endswith('.npz')]

    n_batch = 50
    batched_fnames = [[filenames[start_idxs:start_idxs + n_batch]] for start_idxs in range(0, len(filenames), n_batch)]
    basic_parallel_loop(compute_and_save_signatures_of_books, batched_fnames, n_jobs=25)
