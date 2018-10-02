import torch
import torch.nn as nn
from torchvision import models


class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()

    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)


class CML_Model(nn.Module):
    def __init__(self, args_dict, comments_vocab_size, titles_vocab_size):
        super(CML_Model, self).__init__()

        self.args_dict = args_dict

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = resnet

        # Visual Embedding layer
        self.visual_embedding = nn.Sequential(
            nn.Linear(1000, args_dict.emb_size),
            nn.Tanh(),
        )

        # Comment embedding
        self.text_embedding = nn.Sequential(
            nn.Linear(comments_vocab_size + titles_vocab_size, args_dict.emb_size),
            nn.Tanh(),
        )

        self.table = TableModule()


    def forward(self, img, y1, t1):
        # inputs to the network
        #     - img: image
        #     - y1: comment
        #     - t1: title

        # text embedding
        text_emb = torch.squeeze(self.table([y1, t1],2),1) # joining on the last dim
        text_emb = self.text_embedding(text_emb)
        text_emb = norm(text_emb)

        # visual embedding
        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        # final output: visual embedding and text embedding
        return [visual_emb, text_emb]





