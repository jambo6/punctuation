import sys; sys.path.append('../')
from definitions import *
import numpy as np
from experiments.compute_signatures import compute_group_list_signatures
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class TorchLGBMDataset:
    """ Dataset for loading book sentence signatures for use with LGBM. """
    def __init__(self,
                 labels='author',
                 label_converter=None,
                 depth=3,
                 transform='signature',
                 n_sentence_samples=None,
                 n_jobs=1):
        self.labels = labels
        self.label_converter = label_converter
        self.depth = depth
        self.transform = transform
        self.n_sentence_samples = n_sentence_samples
        self.n_jobs = n_jobs

        self.load_dir = PUNCTUATION_DIR
        self.folders = [x for x in os.listdir(PUNCTUATION_DIR) if x.isdigit()]

    def __len__(self):
        return len(self.folders)

    def get_books(self, book_id):
        return load_pickle(self.load_dir + '/{}/sentences.pkl'.format(book_id))

    def get_alex_features(self, book_id, features=[1, 2, 3, 4, 5, 6]):
        load_f = lambda f: load_pickle(PUNCTUATION_DIR + '/{}/f{}.pkl'.format(book_id, f))
        if isinstance(features, list):
            fs = np.concatenate(
                [load_f(f) for f in features]
            )
        else:
            fs = load_f(features)
        return fs

    def get_labels(self, book_id):
        if self.labels == 'author':
            return load_pickle(self.load_dir + '/{}/author_label.pkl'.format(book_id))
        elif self.labels == 'book':
            return book_id
        else:
            raise NotImplementedError()

    def stacker(self, signatures, labels, idxs):
        """ Stacks copies of labels and ids so they will match the stacked signatures. """
        labels_stacked = torch.cat([torch.Tensor([labels[i]] * signatures[i].size(0)) for i in range(len(signatures))])
        idxs_stacked = torch.cat([torch.Tensor([int(idxs[i])] * signatures[i].size(0)) for i in range(len(signatures))])
        return labels_stacked.numpy().astype(int), idxs_stacked.numpy().astype(int)

    def random_sample_sentences(self, books):
        def sampler(book):
            return [book[i] for i in np.random.randint(0, len(book), self.n_sentence_samples)]
        return [sampler(b) for b in books]

    def __getitem__(self, idxs):
        """ Gets signatures of the books with index ids. """
        # Extract alex data
        alex_features = np.stack([self.get_alex_features(b) for b in idxs])

        # Get books aond resample sentences if specified
        books = [[torch.Tensor(s) for s in self.get_books(idx)] for idx in idxs]
        books = self.random_sample_sentences(books) if isinstance(self.n_sentence_samples, int) else books

        # Drop time and ending punctuation
        ending_punctuation = [torch.argmax(torch.stack([s[-1, [1, 2, 3, 4]] for s in b]), -1).view(-1, 1) for b in books]
        books = [[s[:, 5:] for s in b] for b in books]

        # Lead lag if you want me
        # from experiments.compute_signatures import LeadLag
        # books = [[LeadLag().transform(s.view(-1, s.size(0), s.size(1)))[0] for s in b] for b in books]

        # Extract the signatures
        signatures = compute_group_list_signatures(books, transform=self.transform, depth=self.depth, n_jobs=self.n_jobs)

        # Add ending punctuation onto the signatures
        signatures = [
            torch.cat((ending_punctuation[i].to(signatures[0].dtype), signatures[i]), -1)
            for i in range(len(signatures))
        ]

        # Also get a global signatures
        for book in books:
            for i in range(1, len(book)):
                book[i] += book[i-1][-1, :]


        # Extract the relevant labels
        labels = np.array([self.get_labels(idx) for idx in idxs])

        # Convert the labels
        if self.label_converter is not None:
            labels = np.array([self.label_converter[l] for l in labels])

        # Stack for doing
        labels_stacked, idxs_stacked = self.stacker(signatures, labels, idxs)

        # Return the information as a dictionary
        output_dict = {
            'signatures': signatures,
            'alex_features': alex_features,
            'labels': labels,
            'labels_stacked': labels_stacked,
            'ids': np.array(idxs),
            'ids_stacked': idxs_stacked,
        }

        return output_dict
