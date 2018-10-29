import sys
import numpy as np
import pandas as pd
import sklearn

import torch
import torch.autograd as autograd

from utils import save_obj, load_obj
from text_encodings import get_text_encoding, get_mapped_text

import os.path
from PIL import Image
from torchvision import models, transforms
from sklearn.cross_decomposition import CCA
from sklearn.feature_extraction.text import TfidfTransformer

import argparse


def get_cca_parser():
    parser = argparse.ArgumentParser(description='SemArt CCA model')
    parser.add_argument('--csvtrain', default='SemArt/semart_train.csv', help='Training set data file')
    parser.add_argument('--csvtest', default='SemArt/semart_test.csv', help='Test set data file')
    parser.add_argument('--dir_images', default='SemArt/Images/', help='Path to SemArt images')
    parser.add_argument('--dir_data', default='Data/', help='Path to project data')
    parser.add_argument('--dir_models', default='Models/CCA/', help='Path to models')
    parser.add_argument('--dir_results', default='Results/CCA/', help='Path to models')
    parser.add_argument('--w2i', default='w2i_comments_3k.pckl', help='File to transform words into indices in the comments vocabulary')
    parser.add_argument('--i2w', default='i2w_comments_3k.pckl', help='File to transform indices into words in the comments vocabulary')
    parser.add_argument('--tfidf', default='tfidf_cca.pckl', help='File with the weights of the tfidf model')
    parser.add_argument('--visnet', default='res50', help='Visual network')
    parser.add_argument('--bTrain', default=True, type=bool, help='True to train&test CCA model, False to only test')
    return parser


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return autograd.Variable(x, volatile=volatile)


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def train_cca(args_dict, w2i, visualmodel, transform = None):

    # switch to evaluation mode
    visualmodel.eval()

    # Load training data
    df = pd.read_csv(args_dict.csvtrain, delimiter='\t')
    print('Number of training samples %d' % len(df))

    # Comments into BoW
    print('Get text encodings...')
    ytext = get_mapped_text(df, w2i, field='DESCRIPTION')
    ytitles = get_mapped_text(df, w2i, field='TITLE')

    # Convert to one-hot vector
    y_train = np.zeros((len(ytext), len(w2i)-1), dtype=np.uint8)
    for i, sentence in enumerate(ytext):
        for j, word in enumerate(sentence):
            y_train[i, word-1] += 1

    for i, title in enumerate(ytitles):
        for j, word in enumerate(title):
            y_train[i, word-1] += 1

    # tfidf weights and vectors
    if os.path.exists(args_dict.dir_data + args_dict.tfidf):
        tfidf = load_obj(args_dict.dir_data + args_dict.tfidf)
        y_train_tfidf = tfidf.transform(y_train)
    else:
        tfidf = TfidfTransformer()
        y_train_tfidf = tfidf.fit_transform(y_train)
        save_obj(tfidf, args_dict.dir_data + args_dict.tfidf)
    print('... text encodings done.')

    # Visual encodings
    print('Get visual encodings...')
    imageurls = list(df['IMAGE_FILE'])
    x_train = []
    for idx, imgurl in enumerate(imageurls):

        # Print info
        if idx % 100 == 0:
            print('::::: Image Encoding :::::: %d/%d' % (idx,len(df)))

        # Load and preprocess image
        image = load_image(args_dict.dir_images + imgurl, transform)
        image_tensor = to_var(image, volatile=True)

        # Compute image features
        visualmodel.zero_grad()
        feature = visualmodel(image_tensor)
        x_train.append(feature.data.cpu().numpy())

    x_train = np.squeeze(np.array(x_train), axis=1)
    print('... visual encodings done.')

    # Train CCA
    print('Get CCA transformation...')
    cca = CCA(n_components=128)
    cca.fit(x_train, y_train_tfidf.toarray())
    print('... CCA done.')

    return cca

def test_cca(args_dict, cca, w2i, visualmodel, transform = None):

    # switch to evaluation mode
    visualmodel.eval()

    # Load data
    df = pd.read_csv(args_dict.csvtest, delimiter='\t')
    print('Number of test samples %d' % len(df))

    # Comments into BoW
    print('Get text encodings for test...')
    ytext = get_mapped_text(df, w2i, field='DESCRIPTION')
    ytitles = get_mapped_text(df, w2i, field='TITLE')

    # Convert to one-hot vector
    y_test = np.zeros((len(ytext), len(w2i)-1), dtype=np.uint8)
    for i, sentence in enumerate(ytext):
        for j, word in enumerate(sentence):
            y_test[i, word-1] += 1

    for i, title in enumerate(ytitles):
        for j, word in enumerate(title):
            y_test[i, word-1] += 1

    # tfidf weights and vectors
    tfidf = load_obj(args_dict.dir_data + args_dict.tfidf)
    y_test_tfidf = tfidf.transform(y_test)

    # Visual encodings
    print('Get visual encodings...')
    imageurls = list(df['IMAGE_FILE'])
    x_test = []
    for idx, imgurl in enumerate(imageurls):
        print('::::: Image Encoding :::::: %d/%d' % (idx, len(df)))

        # Load and preprocess image
        imagepath = args_dict.dir_images + imgurl
        image = load_image(imagepath, transform)
        image_tensor = to_var(image, volatile=True)

        # Compute image features
        visualmodel.zero_grad()
        feature = visualmodel(image_tensor)
        x_test.append(feature.data.cpu().numpy())

    x_test = np.squeeze(np.array(x_test), axis=1)
    print('... test visual encodings done.')

    # CCA transformation
    print('Apply CCA transformation...')
    X_c, Y_c = cca.transform(x_test, y_test_tfidf.toarray())

    save_obj(X_c, args_dict.dir_results + 'test_img_embeds.pkl')
    save_obj(Y_c, args_dict.dir_results + 'test_text_embeds.pkl')

    return X_c, Y_c


def evaluation(img_emb, text_emb):

    # Evaluation variables
    med_rank_t2i, med_rank_i2t = [], []
    recall_t2i = {01: 0.0, 05: 0.0, 10: 0.0}
    recall_i2t = {01: 0.0, 05: 0.0, 10: 0.0}
    N = img_emb.shape[0]

    # Measure cosine similarity
    similarities = sklearn.metrics.pairwise.cosine_similarity(text_emb, img_emb)

    # Iterate over each pair
    for idx in range(N):

        # TEXT TO IMAGE
        ranking_t2i = np.argsort(similarities[idx, :])[::-1].tolist()

        # position of idx in ranking
        pos = ranking_t2i.index(idx)
        if (pos + 1) == 1:
            recall_t2i[1] += 1
        if (pos + 1) <= 5:
            recall_t2i[5] += 1
        if (pos + 1) <= 10:
            recall_t2i[10] += 1

        # store the position
        med_rank_t2i.append(pos + 1)

        # IMAGE TO TEXT
        ranking_i2t = np.argsort(similarities[:, idx])[::-1].tolist()

        # position of idx in ranking
        pos2 = ranking_i2t.index(idx)
        if (pos2 + 1) == 1:
            recall_i2t[1] += 1
        if (pos2 + 1) <= 5:
            recall_i2t[5] += 1
        if (pos2 + 1) <= 10:
            recall_i2t[10] += 1

        # store the position
        med_rank_i2t.append(pos2 + 1)

    for i in recall_t2i.keys():
        recall_t2i[i] = recall_t2i[i] / N

    for i in recall_i2t.keys():
        recall_i2t[i] = recall_i2t[i] / N

    print "Median text2image", np.median(med_rank_t2i)
    print "Recall text2image", recall_t2i

    print "Median image2text", np.median(med_rank_i2t)
    print "Recall image2text", recall_i2t


if __name__ == "__main__":

    # Set the correct system encoding to read the csv files
    reload(sys)
    sys.setdefaultencoding('Cp1252')

    # Load parameters
    parser = get_cca_parser()
    args_dict, unknown = parser.parse_known_args()

    # Create directories
    if not os.path.exists(args_dict.dir_results):
        os.makedirs(args_dict.dir_results)
        print('Directory %s created.' %args_dict.dir_results)

    args_dict.cca_model = args_dict.dir_models + args_dict.visnet + '.pckl'
    if not os.path.exists(args_dict.dir_models):
        os.makedirs(args_dict.dir_models)
        print('Directory %s created.' %args_dict.dir_models)

    if not os.path.exists(args_dict.dir_data):
        os.makedirs(args_dict.dir_data)
        print('Directory %s created.' %args_dict.dir_data)

    # Get text encodings
    if os.path.isfile(args_dict.dir_data + args_dict.w2i):
        w2i = load_obj(args_dict.dir_data + args_dict.w2i)
        print('Text encoding loaded from %s.' % args_dict.dir_data + args_dict.w2i)
    else:
        w2i, i2w = get_text_encoding(args_dict.csvtrain, 3000, 'DESCRIPTION')
        save_obj(w2i, args_dict.dir_data + args_dict.w2i)
        save_obj(i2w, args_dict.dir_data + args_dict.i2w)

    # Load visual model
    visionmodel = models.resnet50(pretrained=True)
    if torch.cuda.is_available():
        visionmodel = visionmodel.cuda()

    # Image transformations: 224x224 resize and ImageNet mean substraction
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],
                             std=[0.229, 0.224, 0.225])
    ])

    # Train CCA model
    if args_dict.bTrain:
        cca = train_cca(args_dict, w2i, visionmodel, transform=data_transforms)
        save_obj(cca, args_dict.cca_model)

    # Test CCA model
    cca = load_obj(args_dict.cca_model)
    img_emb, text_emb = test_cca(args_dict, cca, w2i, visionmodel, transform=data_transforms)

    # Measure accuracy and print
    evaluation(img_emb, text_emb)



