import nltk
import itertools
import pandas as pd

from nltk.corpus import stopwords

import numpy as np


def get_text_encoding(fileText, N, field):

    print("Creating text encodings... ")

    # Load training sentences
    df = pd.read_csv(fileText, delimiter='\t')

    # Lower-case comments and convert to list of strings
    comments = []
    for index, row in df.iterrows():
        tokens = nltk.word_tokenize(row[field])
        words = [w.lower() for w in tokens]
        comments.append(words)

    # Get vocabulary
    word_freq = nltk.FreqDist(itertools.chain(*comments))
    vocab = word_freq.most_common(len(word_freq))

    # Get embeddings
    if field == 'DESCRIPTION':
        index_to_word = [x[0] for x in vocab if not x[0] in stopwords.words('english') and x[0].isalpha() and x[1] > 10]
        index_to_word = index_to_word[0:N]
    elif field == 'TITLE':
        index_to_word = [x[0] for x in vocab if not x[0] in stopwords.words('english') and x[0].isalpha()]

    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    print("Vocabulary size : %d." % len(index_to_word))

    print("... Text encodings done.")
    return word_to_index, index_to_word


def get_mapped_text(df, w2i, field):

    sentences = []
    for index, row in df.iterrows():
        tokens = nltk.word_tokenize(row[field])
        words = [w.lower() for w in tokens if w.isalpha()]
        sentences.append(words)

    # Remove OOV tokens from the text
    for i, sent in enumerate(sentences):
        sentences[i] = [w for w in sent if w in w2i]

    # Tokens to vector of indices
    mapped_text = np.asarray([[w2i[w] for w in sent] for sent in sentences])

    return mapped_text
