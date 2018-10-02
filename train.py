import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import sys
import numpy as np
import sklearn

from params import get_parser
from model import CML_Model
from dataloader import SemArtDataset
from text_encodings import get_text_encoding
import utils


def save_checkpoint(state):
    directory = args_dict.dir_model + "%s/"%(args_dict.train_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'model_e%03d_v-%.3f.pth.tar' % (state['epoch'],state['best_val'])
    torch.save(state, filename)


def trainEpoch(train_loader, model, criterion, optimizer, epoch):

    # object to store & plot the losses
    cos_losses = utils.AverageMeter()

    # switch to train mode
    model.train()

    # Train in mini-batches
    for batch_idx, (input, target) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            input_var.append(torch.autograd.Variable(input[j]).cuda())

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            target[j] = target[j].cuda(async=True)
            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        output = model(input_var[0], input_var[1], input_var[2])

        # Compute loss
        train_loss = criterion(output[0], output[1], target_var[0].float())
        cos_losses.update(train_loss.data[0], input[0].size(0))

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'vision ({visionLR}) - comment ({textLR})\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
            loss=cos_losses, visionLR=optimizer.param_groups[1]['lr'], textLR=optimizer.param_groups[0]['lr']))

    # Plot loss after all mini-batches have finished
    plotter.plot('cos_loss', 'train', 'CML Loss', epoch, cos_losses.avg)


def valEpoch(args_dict, val_loader, model, criterion, epoch):

    cos_losses = utils.AverageMeter()

    # switch to evaluation mode
    model.eval()

    # Mini-batches
    for batch_idx, (input, target) in enumerate(val_loader):

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

        # Compute loss
        train_loss = criterion(output[0], output[1], target_var[0].float())
        cos_losses.update(train_loss.data[0], input[0].size(0))

        # Save embeddings to compute rankings later
        if batch_idx==0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[1].data.cpu().numpy()
            data2 = target[-2]
        else:
            data0 = np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
            data1 = np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)
            data2 = np.concatenate((data2,target[-2]),axis=0)

    # Computer MedR and Recall
    medR, recall = rank(args_dict, data0, data1, data2)

    # Print validation info
    print('Validation set: Average loss: {:.4f}\t'
          'medR {medR:.2f}\t'
          'Recall {recall}'.format(cos_losses.avg, medR=medR, recall=recall))

    # Plot validation results
    plotter.plot('cos_loss', 'test', 'Joint Model Loss', epoch, cos_losses.avg)
    plotter.plot('medR', 'test', 'Joint Model medR', epoch, medR)
    plotter.plot('recall', 'test', 'Joint Model Recall at 10', epoch, recall[10])

    # Return MedR as the validation outcome
    return medR


def rank(args_dict, img_embeds, text_embeds, ids):

    # Sort indices
    idxs = np.argsort(ids)
    ind = ids[idxs]
    img_emb = img_embeds[idxs,:]
    text_emb = text_embeds[idxs,:]
    N = len(ind)

    # Accuracy variables
    med_rank = []
    recall = {1: 0.0, 5: 0.0, 10: 0.0}

    # Text to image
    for text, idx in zip(text_emb, ind):

        # Cosine similarities between text and all images
        text = np.expand_dims(text, axis=0)
        similarities = np.squeeze(sklearn.metrics.pairwise.cosine_similarity(text, img_emb), axis=0)
        ranking = np.argsort(similarities)[::-1].tolist()

        # position of idx in ranking
        pos = ranking.index(idx)
        if (pos + 1) == 1:
            recall[1] += 1
        if (pos + 1) <= 5:
            recall[5] += 1
        if (pos + 1) <= 10:
            recall[10] += 1

        # store the position
        med_rank.append(pos + 1)

    for i in recall.keys():
        recall[i] = recall[i] / N

    return np.median(med_rank), recall


def trainProcess(args_dict):

    # Get comments vocabulary
    if os.path.isfile(args_dict.dir_data + args_dict.w2i):
        vocab_comment = utils.load_obj(args_dict.dir_data + args_dict.w2i)
        print('Text encoding loaded from %s' % (args_dict.dir_data + args_dict.w2i))
    else:
        vocab_comment, i2w = get_text_encoding(args_dict.csvtrain, 10000, 'DESCRIPTION')
        utils.save_obj(vocab_comment, args_dict.dir_data + args_dict.w2i)
        utils.save_obj(i2w, args_dict.dir_data + args_dict.i2w)

    # Get titles vocabulary
    if os.path.isfile(args_dict.dir_data + args_dict.w2i_tit):
        vocab_title = utils.load_obj(args_dict.dir_data + args_dict.w2i_tit)
        print('Text encoding loaded from %s' % (args_dict.dir_data + args_dict.w2i_tit))
    else:
        vocab_title, i2w_tit = get_text_encoding(args_dict.csvtrain, -1, 'TITLE')
        utils.save_obj(vocab_title, args_dict.dir_data + args_dict.w2i_tit)
        utils.save_obj(i2w_tit, args_dict.dir_data + args_dict.i2w_tit)

    # Load CML model
    model = CML_Model(args_dict, len(vocab_comment), len(vocab_title))
    if args_dict.use_gpu:
        model.cuda()

    # Loss and optimizer
    cosine_loss = nn.CosineEmbeddingLoss(margin=args_dict.margin).cuda()

    # Create different parameter groups to optimize different parts of the model separately
    vision_params = list(map(id, model.resnet.parameters()))
    base_params   = filter(lambda p: id(p) not in vision_params, model.parameters())

    # Adam Optimizer - with lr initialized accordingly
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.resnet.parameters(), 'lr': args_dict.lr*args_dict.freeVision }
            ], lr=args_dict.lr*args_dict.freeComment)

    # Resume training if needed
    # best_val is measured in terms of MedR, so the lower the better
    if args_dict.resume:
        if os.path.isfile(args_dict.resume):
            print("=> loading checkpoint '{}'".format(args_dict.resume))
            checkpoint = torch.load(args_dict.resume)
            args_dict.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args_dict.resume, checkpoint['epoch']))
            best_val = float('inf')
        else:
            print("=> no checkpoint found at '{}'".format(args_dict.resume))
            best_val = float('inf')
    else:
        best_val = float('inf')

    # Data transformation for training (with data augmentation) and validation
    train_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.RandomHorizontalFlip(),                  # random horizontal flip (data augmentation)
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(
        SemArtDataset(args_dict,
                   set = 'train',
                   w2i_tit = vocab_title,
                   w2i = vocab_comment,
                   transform = train_transforms),
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Training loader with %d samples' % train_loader.__len__())

    val_loader = torch.utils.data.DataLoader(
        SemArtDataset(args_dict,
                   set = 'val',
                   w2i_tit=vocab_title,
                   w2i=vocab_comment,
                   transform = val_transforms),
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Validation loader with %d samples' % val_loader.__len__())

    # Now, let's start the training process!
    print('Training...')
    valtrack = 0 # measures patience
    for epoch in range(args_dict.start_epoch, args_dict.nepochs):

        # Compute a training epoch
        trainEpoch(train_loader, model, cosine_loss, optimizer, epoch)

        # Compute a validation epoch
        lossval = valEpoch(args_dict, val_loader, model, cosine_loss, epoch)

        # check patience
        if lossval >= best_val:
            valtrack += 1
        else:
            valtrack = 0

        # if patience, switch params update
        if valtrack >= args_dict.patience:
            args_dict.freeVision = args_dict.freeComment
            args_dict.freeComment = not (args_dict.freeVision)
            optimizer.param_groups[0]['lr'] = args_dict.lr * args_dict.freeComment
            optimizer.param_groups[1]['lr'] = args_dict.lr * args_dict.freeVision
            print 'Initial base params lr: %f' % optimizer.param_groups[0]['lr']
            print 'Initial vision lr: %f' % optimizer.param_groups[1]['lr']
            args_dict.patience = 3
            valtrack = 0

        # save if it is the best model
        is_best = lossval < best_val
        best_val = min(lossval, best_val)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': valtrack,
                'freeVision': args_dict.freeVision,
                'curr_val': lossval,
            })
        print '** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack)


if __name__ == "__main__":

    # Set the correct system encoding to read the csv files
    reload(sys)
    sys.setdefaultencoding('Cp1252')

    # Load parameters
    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    # Set seed for reproducibility
    torch.manual_seed(args_dict.seed)
    if args_dict.use_gpu:
        torch.cuda.manual_seed(args_dict.seed)

    # Plots
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=args_dict.train_name)

    # Training process
    trainProcess(args_dict)