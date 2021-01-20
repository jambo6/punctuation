from definitions import *
import numpy as np
import pandas as pd


def load_book_ids():
    """ Given a list of book ids, loads them in. """
    pass


if __name__ == '__main__':
    book_to_author_label = np.load(
        DATA_DIR + '/processed/Periodic/Punctuation/other/book_to_author_label.pkl', allow_pickle=True
    )
    book_to_author = pd.DataFrame.from_dict(data=book_to_author_label, orient='index').value_counts()