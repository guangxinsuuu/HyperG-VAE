import scipy.sparse as sp  # Sparse matrix operations
import numpy as np  # Numerical computations
from sklearn.metrics import precision_recall_curve, roc_curve  # Metrics for evaluation
import pandas as pd  # Data handling
import copy  # For deep copying objects
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


def normalize_sparse_hypergraph_symmetric(H):
    """
    Normalize a sparse hypergraph using symmetric normalization.
    
    Args:
        H (scipy.sparse matrix): Hypergraph incidence matrix.

    Returns:
        scipy.sparse matrix: Normalized symmetric hypergraph Laplacian.
    """
    # Row normalization (degree of nodes)
    rowsum = np.array(H.sum(1))  # Sum of each row
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # Compute D^(-1/2)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.  # Handle infinite values
    D = sp.diags(r_inv_sqrt)  # Create diagonal matrix for row normalization

    # Column normalization (degree of hyperedges)
    colsum = np.array(H.sum(0))  # Sum of each column
    r_inv_sqrt = np.power(colsum, -1).flatten()  # Compute B^(-1)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.  # Handle infinite values
    B = sp.diags(r_inv_sqrt)  # Create diagonal matrix for column normalization

    # Construct the normalized Laplacian
    Omega = sp.eye(B.shape[0])  # Identity matrix for hyperedge weighting
    hx = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)

    return hx


def extractEdgesFromMatrix(m, geneNames, TFmask):
    """
    Extract edges from an adjacency matrix, optionally filtering by a transcription factor (TF) mask.
    
    Args:
        m (numpy.ndarray): Adjacency matrix (edge weights).
        geneNames (list): List of gene names corresponding to nodes.
        TFmask (numpy.ndarray): Optional mask for TF-target relationships.
    
    Returns:
        pandas.DataFrame: DataFrame containing edges with columns ['TF', 'Target', 'EdgeWeight'].
    """
    geneNames = np.array(geneNames)  # Convert gene names to a NumPy array
    mat = copy.deepcopy(m)  # Deep copy of the adjacency matrix
    num_nodes = mat.shape[0]
    
    # Initialize indicator matrix for edges
    mat_indicator_all = np.zeros([num_nodes, num_nodes])

    # Apply TF mask if provided
    if TFmask is not None:
        mat = mat * TFmask

    # Identify edges with non-zero weights
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)  # Get indices of edges

    # Create a DataFrame for edges
    edges_df = pd.DataFrame({
        'TF': geneNames[idx_send],  # Source (TF)
        'Target': geneNames[idx_rec],  # Target
        'EdgeWeight': (mat[idx_rec, idx_send])  # Edge weight
    })

    # Sort edges by weight in descending order
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)

    return edges_df


def evaluate(A, truth_edges, Evaluate_Mask):
    """
    Evaluate predicted edges against ground truth edges.
    
    Args:
        A (numpy.ndarray): Predicted adjacency matrix.
        truth_edges (set): Ground truth edges as a set of tuples (source, target).
        Evaluate_Mask (numpy.ndarray): Mask to restrict evaluation to specific edges.
    
    Returns:
        tuple: Number of overlapping edges, enrichment precision rate (EPR).
    """
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)

    # Take absolute values for edge weights
    A = abs(A)

    # If no mask provided, consider all edges except self-loops
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))

    # Apply evaluation mask
    A = A * Evaluate_Mask

    # Rank edges by weight
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0][::-1])  # Flatten and sort in descending order
    cutoff_all = A_val[num_truth_edges]  # Determine threshold for top-k edges

    # Predict edges based on threshold
    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    idx_rec, idx_send = np.where(A_indicator_all)  # Get predicted edge indices
    A_edges = set(zip(idx_send, idx_rec))  # Predicted edges as a set

    # Find overlapping edges between prediction and ground truth
    overlap_A = A_edges.intersection(truth_edges)

    # Enrichment Precision Rate (EPR)
    EPR = 1.0 * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))

    return len(overlap_A), EPR
