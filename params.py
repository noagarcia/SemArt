import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Text2Art challenge')
    parser.add_argument('--train_name', default = 'CML', help='Name')

    # Directories
    parser.add_argument('--dir_images', default='SemArt/Images/', help='Path to SemArt images')
    parser.add_argument('--dir_data', default='Data/', help='Path to project data')
    parser.add_argument('--dir_model', default='Models/', help='Path to project data')

    # Vocabularies
    parser.add_argument('--w2i', default='w2i_comments_10k.pckl', help='File to transform words into indices in the comments vocabulary')
    parser.add_argument('--i2w', default='i2w_comments_10k.pckl', help='File to transform indices into words in the comments vocabulary')
    parser.add_argument('--w2i_tit', default='w2i_titles.pckl', help='File to transform words into indices in the titles vocabulary')
    parser.add_argument('--i2w_tit', default='i2w_titles.pckl', help='File to transform indices into words in the titles vocabulary')
    parser.add_argument('--tfidf_coms_file', default='tfidf_coms_cml.pckl', help='File with the weights of the tfidf model')
    parser.add_argument('--tfidf_tits_file', default='tfidf_tits_cml.pckl', help='File with the weights of the tfidf model')

    # SemArt csv files
    parser.add_argument('--csvtrain', default='SemArt/semart_train.csv', help='Training set data file')
    parser.add_argument('--csvval', default='SemArt/semart_val.csv', help='Dataset val data file')
    parser.add_argument('--csvtest', default='SemArt/semart_test.csv', help='Dataset test data file')

    # Training
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=123, help='Seed for reproducibility', type=int)
    parser.add_argument('--use_gpu', default=True, help='Use gpu', type=bool)
    parser.add_argument('--batch_size', default=28, help='Batch size', type=int)
    parser.add_argument('--lr', default=0.0001, help='Learning rate', type=float)
    parser.add_argument('--freeVision', default=False, type=bool)
    parser.add_argument('--freeComment', default=True, type=bool)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, help='Initial epoch', type=int)
    parser.add_argument('--nepochs', default=120, help='Number of epochs', type=int)

    # CML Model params
    parser.add_argument('--margin', default=0.1, help='Loss margin', type=float)
    parser.add_argument('--emb_size', default=128, help='Embedding size', type=int)
    parser.add_argument('--patience', default=1, type=int)

    # Test
    parser.add_argument('--model_path', default='Models/CML/model_e024_v-10.000.pth.tar', type=str)
    parser.add_argument('--path_results', default='Results/CML/', type=str)
    parser.add_argument('--no_cuda', action='store_true')

    return parser