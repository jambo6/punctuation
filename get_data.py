from definitions import load_pickle, save_pickle
import os
import urllib.request
DATA_DIR = './data'


def download():
    assert os.path.isdir(DATA_DIR + '/raw/Punctuation'), "No directory exists at data/raw/Punctuation. " \
                                                         "Please make one to continue."
    # Download
    filename = DATA_DIR + '/raw/Punctuation/punctuation.pkl'
    url = 'https://zenodo.org/record/3605100/files/punctuation_stylometry.p?download=1'
    if os.path.isfile(filename):
        print('File already exists at {}, delete to redownload.'.format(filename))
    else:
        urllib.request.urlretrieve(url, filename)


def make_reduced_set():
    frame = load_pickle(DATA_DIR + '/raw/Punctuation/punctuation.pkl')
    save_pickle(frame.iloc[0:100], DATA_DIR + '/raw/Punctuation/punctuation_reduced.pkl')


if __name__ == '__main__':
    download()
    make_reduced_set()

