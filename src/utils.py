
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
import pandas as pd
import copy
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
def normalize_sparse_hypergraph_symmetric(H):
    
        rowsum = np.array(H.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        D = sp.diags(r_inv_sqrt)
        
        colsum = np.array(H.sum(0))
        r_inv_sqrt = np.power(colsum, -1).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        B = sp.diags(r_inv_sqrt)
        
        Omega = sp.eye(B.shape[0])

        hx = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)

        return hx

def extractEdgesFromMatrix(m, geneNames,TFmask):
    geneNames = np.array(geneNames)
    mat = copy.deepcopy(m)
    num_nodes = mat.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        mat = mat*TFmask
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'TF': geneNames[idx_send], 'Target': geneNames[idx_rec], 'EdgeWeight': (mat[idx_rec, idx_send])})
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)

    return edges_df


def evaluate(A, truth_edges, Evaluate_Mask):
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)
    A= abs(A)
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0][::-1])
    # A_val.reverse()
    cutoff_all = A_val[num_truth_edges]
    #Predicted P
    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)
    A_edges = set(zip(idx_send, idx_rec))
   
    overlap_A = A_edges.intersection(truth_edges)
   
    return len(overlap_A), 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask)),





