import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
import pandas as pd
from text_encodings import get_mapped_text
from sklearn.feature_extraction.text import TfidfTransformer
from utils import load_obj, save_obj


class SemArtDataset(data.Dataset):

    def __init__(self, args_dict, set, w2i_tit, w2i, transform = None):
        """
        Args:
            set: 'train', 'val', 'test
            w2i_tit: word to index for titles
            w2i: word to index for comments
            transform: data transform
        """
        self.args_dict = args_dict
        self.set = set

        # Load Data
        if self.set == 'train':
            textfile = args_dict.csvtrain
            self.mismtch = 0.8
        elif self.set == 'val':
            textfile = args_dict.csvval
            self.mismtch = 0
        elif self.set == 'test':
            textfile = args_dict.csvtest
            self.mismtch = 0
        df = pd.read_csv(textfile, delimiter='\t')
        self.imageurls = list(df['IMAGE_FILE'])
        self.comment_map = get_mapped_text(df, w2i, field='DESCRIPTION')
        self.titles_map = get_mapped_text(df, w2i_tit, field='TITLE')

        # Parameters
        self.numpairs = len(df) / (1 - self.mismtch)
        self.comw2i = w2i
        self.titw2i = w2i_tit
        # self.titw2i = dict([(w, i) for i, w in enumerate(titvocab)])
        self.imagefolder = args_dict.dir_images
        self.transform = transform

        # tfidf weights and vectors
        if os.path.exists(args_dict.dir_data + args_dict.tfidf_coms_file):
            self.tfidf_coms = load_obj(args_dict.dir_data + args_dict.tfidf_coms_file)
        else:
            self.tfidf_coms = self.get_tfidf(self.comment_map, self.comw2i)
            save_obj(self.tfidf_coms, args_dict.dir_data + args_dict.tfidf_coms_file)

        if os.path.exists(args_dict.dir_data + args_dict.tfidf_tits_file):
            self.tfidf_tits = load_obj(args_dict.dir_data + args_dict.tfidf_tits_file)
        else:
            self.tfidf_tits = self.get_tfidf(self.titles_map, self.titw2i)
            save_obj(self.tfidf_tits, args_dict.dir_data + args_dict.tfidf_tits_file)


    def get_tfidf(self, text_map, w2i):
        """Computes TFIDF weights for text encoding"""

        # One-hot vectors
        text_onehot = np.zeros((len(text_map),len(w2i)), dtype=np.uint8)
        for i, sentence in enumerate(text_map):
            for j, word in enumerate(sentence):
                text_onehot[i, word] += 1

        # TFIDF computation
        tfidf = TfidfTransformer()
        tfidf.fit_transform(text_onehot)

        # Return
        return tfidf


    def __len__(self):
        """Return the length of dataset, which is the number of pairs"""
        return self.numpairs


    def __getitem__(self, index):
        """Returns data sample as a pair (image, description)."""

        # Pick comment/attributes index --> idx_text
        idx_text = index % len(self.imageurls)

        # Assign if pair is a match or non-match --> target
        if self.set == 'train':
            match = np.random.uniform() > self.mismtch
        else:
            match = True
        target = match and 1 or -1

        # Pick image index: same as idx_text if match, random if non-match --> idx_img
        if target == 1:
            idx_img = idx_text
        else:
            all_idx = range(len(self.imageurls))
            idx_img = np.random.choice(all_idx)
            while idx_img == idx_text:
                idx_img = np.random.choice(all_idx)

        # Load idx_img image & apply transformation --> image
        imagepath = self.imagefolder + self.imageurls[idx_img]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Encode idx_text comment as a tfidf vector --> comment
        comm_map = self.comment_map[idx_text]
        comm_onehot = np.zeros((len(self.comw2i)), dtype=np.uint8)
        for word in comm_map:
            comm_onehot[word] += 1
        comm_tfidf = self.tfidf_coms.transform(comm_onehot)
        comment = torch.FloatTensor(comm_tfidf.toarray())

        # Encode idx_text title as a tfidf vector --> title
        tit_map = self.titles_map[idx_text]
        tit_onehot = np.zeros((len(self.titw2i)), dtype=np.uint8)
        for word in tit_map:
            tit_onehot[word] += 1
        tit_tfidf = self.tfidf_tits.transform(tit_onehot)
        title = torch.FloatTensor(tit_tfidf.toarray())

        # Return
        if self.set == 'train':
            return [image, comment, title], [target]
        else:
            return [image, comment, title], [target, idx_img, idx_text]