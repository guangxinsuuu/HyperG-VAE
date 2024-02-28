import scanpy
import argparse
from src.UMAP_model import test_non_celltype_GRN_model
import torch
import numpy as np
parser = argparse.ArgumentParser()
#添加random seed
parser.add_argument('--n_epochs', type=int, default=60, help='Number of Epochs for training')
parser.add_argument('--task', type=str, default='non_celltype_GRN',
                    help='Determine which task to run. Select from (non_celltype_GRN,celltype_GRN)')
parser.add_argument('--setting', type=str, default='test', help='Determine whether or not to use the default hyper-parameter')
# parser.add_argument('--batch_size', type=int, default=3902, help='The batch size used in the training process.')

parser.add_argument('--nalpha', type=int, default=1, help='The loss coefficient for L1 norm of W, which is same as \\alpha used in our paper.')
parser.add_argument('--nbeta', type=int, default=1, help='The loss coefficient for KL term (beta-VAE), which is same as \\beta used in our paper.')
parser.add_argument('--nomega', type=int, default=1, help='The loss coefficient for KL term (beta-VAE), which is same as \\beta used in our paper.')
parser.add_argument('--nlr', type=int, default=5, help='The learning rate of used for RMSprop.')
parser.add_argument('--nlr_step_size', type=int, default=2, help='The step size of learning rate decay.')
parser.add_argument('--gamma', type=float, default=0.95, help='The decay factor of learning rate')
parser.add_argument('--n_hidden', type=int, default=128, help='The Number of hidden neural used in MLP')
parser.add_argument('--K', type=int, default=1, help='Number of Gaussian kernel in GMM, default =1')
parser.add_argument('--K1', type=int, default=1, help='The Number of epoch for optimize MLP. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
parser.add_argument('--K2', type=int, default=2, help='The Number of epoch for optimize W. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
parser.add_argument('--save_name', type=str, default='tempbcell')
parser.add_argument('--seed', type=int, default=9, help='seed, default =1')
parser.add_argument('--heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropoutrate', type=float, default=5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset_name', type=str, default='WT2LD')
#parser.add_argument('--data_file', type=str, default='demo_data\embedding\input\Zeisel.csv',help='The input scRNA-seq gene expression file.')
#parser.add_argument('--batch_size', type=int, default=64, help='The batch size used in the training process.')

if parser.parse_args().dataset_name == "zeisel":
    parser.add_argument('--data_file', type=str, default='demo_data\GRN_inference\inputdata\Zeisel.csv',help='The input scRNA-seq gene expression file.')
    parser.add_argument('--net_file', type=str, default='demo_data\GRN_inference\inputdata\ChipNon_500_mHSCGM\label.csv',
                    help='The ground truth of GRN. Only used in GRN inference task if available. ')

#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168158
if parser.parse_args().dataset_name == "WTLD":
    parser.add_argument('--data_file', type=str, default='demo_data\GRN_inference\inputdata\WTLD.csv',help='The input scRNA-seq gene expression file.')
    parser.add_argument('--net_file', type=str, default='demo_data\GRN_inference\inputdata\ChipNon_500_mHSCGM\label.csv',
                    help='The ground truth of GRN. Only used in GRN inference task if available. ')
    parser.add_argument('--batch_size', type=int, default=7454, help='The batch size used in the training process.')

if parser.parse_args().dataset_name == "WT2_RagKappaPreB":
    parser.add_argument('--data_file', type=str, default='demo_data\GRN_inference\inputdata\WT2_RagKappaPreB.csv',help='The input scRNA-seq gene expression file.')
    parser.add_argument('--net_file', type=str, default='demo_data\GRN_inference\inputdata\ChipNon_500_mHSCGM\label.csv',
                    help='The ground truth of GRN. Only used in GRN inference task if available. ')



opt = parser.parse_args()
opt.alpha = 0.1*10**(opt.nalpha)
opt.beta =  0.1*opt.nbeta
opt.dropout = 0.1*opt.dropoutrate
opt.omega = 0.1*opt.nomega
opt.lr = 1*10**(-opt.nlr)
opt.lr_step_size = 1*10**(-opt.nlr_step_size)
seed = opt.seed
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    if opt.task == 'non_celltype_GRN':
        
        if opt.setting == 'test':
            # opt.beta = 1
            # opt.alpha = 100
            # opt.K1 = 1
            # opt.K2 = 2
            # opt.n_hidden = 128
            # opt.gamma = 0.95
            # opt.lr = 1e-4
            # opt.lr_step_size = 0.99
            # opt.batch_size = 64
            model = test_non_celltype_GRN_model(opt)
            
        model.train_model()
  
