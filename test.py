import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
import numpy as np
import sklearn

import torch
from torchvision import transforms

from model import CML_Model
from dataloader import SemArtDataset
from params import get_parser
import utils


def measure_test_acc(img_emb, text_emb, text_ind):

    # Sort indices
    idxs = np.argsort(text_ind)
    text_ind = text_ind[idxs]
    img_emb = img_emb[idxs,:]
    text_emb = text_emb[idxs,:]
    N = len(text_ind)

    # Accuracy variables
    med_rank_t2i, med_rank_i2t = [], []
    recall_t2i = {1: 0.0, 5: 0.0, 10: 0.0}
    recall_i2t = {1: 0.0, 5: 0.0, 10: 0.0}

    # Text to image
    for text, idx in zip(text_emb, text_ind):

        # Cosine similarities between text and all images
        text = np.expand_dims(text, axis=0)
        similarities = np.squeeze(sklearn.metrics.pairwise.cosine_similarity(text, img_emb), axis=0)
        ranking_t2i = np.argsort(similarities)[::-1].tolist()

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

    # Image to text
    for img, idx in zip(img_emb, text_ind):

        # Cosine similarities between text and all images
        img = np.expand_dims(img, axis=0)
        similarities = np.squeeze(sklearn.metrics.pairwise.cosine_similarity(img, text_emb), axis=0)
        ranking_i2t = np.argsort(similarities)[::-1].tolist()

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

    print "Median t2i", np.median(med_rank_t2i)
    print "Recall t2i", recall_t2i

    print "Median i2t", np.median(med_rank_i2t)
    print "Recall i2t", recall_i2t


def extract_test_features(args_dict):

    # Get comments vocabulary
    vocab_comment = utils.load_obj(args_dict.dir_data + args_dict.w2i)
    vocab_title = utils.load_obj(args_dict.dir_data + args_dict.w2i_tit)

    # Load CML model
    model = CML_Model(args_dict, len(vocab_comment), len(vocab_title))
    if args_dict.use_gpu:
        model.cuda()

    # Load best model
    print("=> loading checkpoint '{}'".format(args_dict.model_path))
    checkpoint = torch.load(args_dict.model_path)
    args_dict.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args_dict.model_path, checkpoint['epoch']))

    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    test_loader = torch.utils.data.DataLoader(
        SemArtDataset(args_dict,
                   set = 'test',
                   w2i_tit=vocab_title,
                   w2i=vocab_comment,
                   transform = test_transforms),
        batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)

    # Switch to evaluation mode & compute test samples embeddings
    model.eval()
    for i, (input, target) in enumerate(test_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j], volatile=True).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j], volatile=True))

        # Output of the model
        output = model(input_var[0], input_var[1], input_var[2])

        # Store embeddings
        if i==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[1].data.cpu().numpy()
            data2 = target[-2]
        else:
            data0 = np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
            data1 = np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)
            data2 = np.concatenate((data2,target[-2]),axis=0)

    # Save embeddings
    utils.save_obj(data0, args_dict.path_results + 'img_embeds.pkl')
    utils.save_obj(data1, args_dict.path_results + 'text_embeds.pkl')
    utils.save_obj(data2, args_dict.path_results + 'ids.pkl')

    return data0, data1, data2


if __name__ == "__main__":

    reload(sys)
    sys.setdefaultencoding('Cp1252')

    # Set the correct system encoding to read the csv files
    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    # Create directories
    if not os.path.exists(args_dict.path_results):
        os.makedirs(args_dict.path_results)

    # Extract features
    img_emd, text_emd, ind = extract_test_features(args_dict)

    # Measure accuracy
    measure_test_acc(img_emd, text_emd, ind)