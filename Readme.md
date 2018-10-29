


# SemArt Project

This is the Pytorch code for the paper "How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval", where we study automatic art interpretation via multi-modal retrieval. Given a set of artistic comments and fine-art paintings, we encode texts and images into a common semantic space, so that comments and paintings that are semantically relevant are encoded close to each other.

![info](https://github.com/noagarcia/SemArt/blob/master/info/overview.png?raw=true)

## Prerequisits
- Python 2.7
- PyTorch (tested with 0.4.0) and torchvision (tested with 0.2.1)
- Python packages: scipi, numpy, sklearn, pandas, PIL, pickle
- [visdom][2]


## Dataset

We introduced the SemArt, a multi-modal dataset for semantic art understanding. SemArt is a collection of fine-art painting images in which each image is associated to a number of attributes and a textual artistic comment, such as those that appear in art catalogues or museum collections. SemArt can be downloaded from [here][1].

![info](https://github.com/noagarcia/SemArt/blob/master/info/sample.png?raw=true)

## Text2Art Challenge

To evaluate semantic art understanding, we propose the Text2Art challenge, a multi-modal retrieval task where relevant paintings are retrieved according to an artistic text, and vice versa. We evaluate several models on this task:

![info](https://github.com/noagarcia/SemArt/blob/master/info/models.png?raw=true)

### CCA 
CCA baseline in which image and text encodings are mapped into a common semantic space using a linear transformation by maximazing correlations between the projected vectors. 

To train and test CCA model:

```
python cca.py
```

### CML
Cosine Margin Loss model in which image and text encodings are mapped into a common semantic space by using pairs of similar and dissimilar samples and optimizing a non-linear transformation with a cosine margin loss function.

To tain CML code:
```
python train.py
```

To test CML model:
```
python train.py
```

Note: parameters can be changed in ```params.py``` file.


### AMD
Coming soon.

## Results

Text2image results:

| Model        | R@1           | R@5  |    R@10    | MedR |
| ------------- |:-------------:| -----:|---------:|--------:|
| CCA | 0.117 | 0.283 | 0.377 | 25 |
| CML | 0.164 | 0.386 | 0.505 | 10 | 

Image2text results:

| Model        | R@1           | R@5  |    R@10    | MedR |
| ------------- |:-------------:| -----:|---------:|--------:|
| CCA | 0.131 | 0.280 | 0.355 | 26 |
| CML | 0.162 | 0.366 | 0.479 | 12 |



## Citation

If using this code, please cite:

```
@article{garcia2018how,
   author    = {Noa Garcia and George Vogiatzis},
   title     = {How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval},
   booktitle = {Proceedings of the European Conference in Computer Vision Workshops},
   year      = {2018},
}
``` 

[1]: http://researchdata.aston.ac.uk/380/
[2]: https://github.com/facebookresearch/visdom
