import sys; sys.path.append('../')
from definitions import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
import signatory


def convert_to_array(item_list):
    """Converts the list of word counts and punctuation onto an array.

    Args:
        item_list (list): A list of ints and strings.

    Returns:
        np.array: The array format of the data where each column corresponds to a punctuation mark or number of words.
    """
    # For conversion
    conversion_dict = {
        # 'words': 0
        '.': 1,
        '!': 2,
        '?': 3,
        '^': 4,
        ':': 5,
        ';': 6,
        '(': 7,
        ')': 8,
        ',': 9,
        '"': 10,
    }

    # Create an array to store all the info
    arr = np.zeros(shape=(int(len(item_list)/2), 11))
    for i, item in enumerate(item_list):
        idx = int(np.floor(i / 2))
        if isinstance(item, int):
            arr[idx, 0] = item
        else:
            arr[idx, conversion_dict[item]] = 1

    return arr.astype(int)


def convert_to_binary_array(arr):
    def dec_to_bin(x):
        bin = np.array([int(x) for x in list('{0:04b}'.format(x))])
        return bin
    binary = np.array([dec_to_bin(x) for x in np.argwhere(arr == 1)[:, 1]])
    return binary


def array_to_sentence_list(arr, cumsum=True, add_time=True, append_zero=True):
    """Splits the array of punctuation marks (and words) into a list of sentences with corresponding sentence ender.

    We assume `arr` is an array such that the first 4 channels correspond to a binary marker of '.', '!', '?', '^'. We
    split the array into sentences whenever such a marker is hit and add the sequence into a list. THe final list is a
    list of tuples of the form `(sentence_end_idx, sentence_arr)` where `sentence_end_idx` marks is 0, 1, 2, 3 for '.',
    '!', '?', '^', respectively.
    """
    # Get .!? locations
    sentence_ends = (arr[:, [0, 1, 2, 3]] == 1).sum(axis=1)
    end_idxs = np.argwhere(sentence_ends).reshape(-1)
    ending_punctuation = np.argmax(arr[end_idxs][:, [0, 1, 2, 3]], axis=1)
    # Only include to last .!?^
    arr_ = arr[:end_idxs[-1] + 1]
    ending_punctuation_list, sentences, start = [], [], 0
    for i in range(len(end_idxs)):
        end = end_idxs[i]
        new_arr = arr_[start:end+1]
        if append_zero:
            new_arr = np.concatenate([np.zeros(new_arr.shape[-1]).reshape(1, -1), new_arr], 0)
        if cumsum:
            new_arr = np.cumsum(new_arr, 0)
        if add_time:
            new_arr = np.concatenate([np.arange(len(new_arr)).reshape(-1, 1), new_arr], 1)
        sentences.append(new_arr)
        ending_punctuation_list.append(ending_punctuation[i])
        start = end + 1
    as_frame = pd.DataFrame(columns=['ending_punctuation'], data=ending_punctuation_list)
    as_frame['sentences'] = sentences
    return as_frame


def get_alexandra_features(frame):
    # Extractor functions
    N = len(frame)
    series_list_to_array = lambda col: np.array([np.array(x) for x in frame[col].values])
    series_mat_to_array = lambda col: np.stack(frame[col].values).reshape(N, -1)
    # Get features
    f1 = series_list_to_array('freq_pun')
    f2 = series_mat_to_array('tran_mat')
    f3 = series_mat_to_array('normalised_tran_mat')
    f4 = series_list_to_array('freq_length_sen')
    f5 = series_list_to_array('freq_word_nb_punctuation')
    f6 = series_mat_to_array('mat_nb_words')
    return f1, f2, f3, f4, f5, f6


def create_stacked_books(books):
    """ Stacks the sentences in each book into a single array being careful to preserve the cumulative sum. """
    stacked_books = []
    for book in books:
        stacked_book = []
        for i in range(len(book)):
            if i == 0:
                stacked_book.append(book[i])
            else:
                update = book[i] + stacked_book[i - 1][-1]
                stacked_book.append(update)
        stacked_books.append(np.concatenate(stacked_book))
    return stacked_books


def extract_and_save_individual_books():
    # Load
    filename = '../data/raw/Periodic/Punctuation/punctuation.pkl'
    frame_full = load_pickle(filename)

    # Different saves
    # Top 10 saves the top 10 most common authors
    other_dir = '../data/processed/Periodic/Punctuation/other'
    author_occurrence_ranking = frame_full['author'].value_counts()
    save_pickle(author_occurrence_ranking, other_dir + '/author_occurrence_ranking.pkl')

    # Convert labels to ints
    le = LabelEncoder()
    labels = frame_full['author']
    # labels_int = le.fit_transform(labels)
    author_to_label = {le.classes_[i]: i for i in range(len(le.classes_))}
    save_pickle(author_to_label, other_dir + '/author_to_label.pkl')

    # Create author to book dict
    authors, book_ids = frame_full['author'].values, frame_full['book_id'].values
    author_to_books = {}
    for author in np.unique(authors):
        mask = authors == author
        author_to_books[author] = book_ids[mask].astype(int)
    save_pickle(author_to_books, other_dir + '/author_to_book_id.pkl')

    # Book to author dict
    book_to_author = {b: a for b, a in zip(book_ids, authors)}
    save_pickle(book_to_author, other_dir + '/book_to_author.pkl')

    # Book to author label
    book_to_author_label = {b: author_to_label[a] for b, a in zip(book_ids, authors)}
    save_pickle(book_to_author_label, other_dir + '/book_to_author_label.pkl')

    # Save general information
    general_columns = ['book_id', 'title', 'author', 'author_birthdate', 'author_deathdate', 'genre', 'language', 'subject']
    general_frame = frame_full[general_columns]
    save_pickle(general_frame, other_dir + '/info_frame.pkl')

    n_batch = 5
    for start_idx in tqdm(range(0, len(frame_full), n_batch)):
        # get books
        frame = frame_full.iloc[start_idx:start_idx + n_batch]

        # Get corresponding alexandra features
        f1, f2, f3, f4, f5, f6 = get_alexandra_features(frame)
        all_alexandra_features = np.concatenate([f1, f2, f3, f4, f5, f6], axis=1)

        # Add an array seq col
        arr_seqs = frame['seq_nb_words'].apply(convert_to_array).values
        word_seqs = [a[:, 0] for a in arr_seqs]
        punc_seqs = [a[:, 1:] for a in arr_seqs]

        # Convert to binary punctuation
        binary_punc_seqs = [convert_to_binary_array(x) for x in punc_seqs]

        # Get book ids
        ids = frame['book_id'].values.astype(int)
        authors = frame['author'].values

        # Save everything
        for ix, id in enumerate(ids):
            np.savez(
                DATA_DIR + '/processed/Periodic/Punctuation/{}.npz'.format(id),
                words_sequence=word_seqs[ix],
                punctuation_sequence=punc_seqs[ix],
                binary_punctuation_sequence=binary_punc_seqs[ix],
                author_label=author_to_label[authors[ix]],
                book_id=ids.tolist()[ix],
                all_alexandra_features=all_alexandra_features[ix],
                f1=f1[ix],
                f2=f2[ix],
                f3=f3[ix],
                f4=f4[ix],
                f5=f5[ix],
                f6=f6[ix],
            )


if __name__ == '__main__':
    extract_and_save_individual_books()
