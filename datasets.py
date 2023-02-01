
import numpy as np 

def load_data(params):

    if params.data_name in ['cite']:
        x1 = np.load('data/cite-seq/cite_data1.npy')
        x2 = np.load('data/cite-seq/cite_data2.npy')
        y1 = np.load('data/cite-seq/cite_labels1.npy')
        y2 = np.load('data/cite-seq/cite_labels2.npy')

        X_list = [x1, x2]
        Y_list = [y1, y2]

    elif params.data_name in ['cite_sample']:
        data_sample = np.load("data/cite-seq/cite_sample.npy", allow_pickle=True).item()
        x1 = data_sample['data'][0]
        x2 = data_sample['data'][1]

        y1 = data_sample['label'][0]
        y2 = data_sample['label'][1]

        X_list = [x1, x2]
        Y_list = [y1, y2]

    elif params.data_name in ['scGEM']:
        x1 = np.loadtxt("data/scGEM/GeneExpression.txt")
        x2 = np.loadtxt("data/scGEM/DNAmethylation.txt")
        y1 = np.loadtxt("data/scGEM/type1.txt").astype(np.int)
        y2 = np.loadtxt("data/scGEM/type2.txt").astype(np.int)

        X_list = [x1, x2]
        Y_list = [y1, y2]

    elif params.data_name in ['SNARE']:
        x1 = np.load("data/SNARE/scatac_feat.npy")
        x2 = np.load("data/SNARE/scrna_feat.npy")
        y1 = np.loadtxt("data/SNARE/SNAREseq_atac_types.txt").astype('int')
        y2 = np.loadtxt("data/SNARE/SNAREseq_rna_types.txt").astype('int')

        X_list = [x1, x2]
        Y_list = [y1, y2]

    return X_list, Y_list