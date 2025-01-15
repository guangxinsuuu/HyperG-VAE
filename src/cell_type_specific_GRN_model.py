# Import required libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc  # Library for single-cell data analysis
import torch.optim as optim  # Optimizers for training
from torch.autograd import Variable  # Enables automatic differentiation
from torch.utils.data import DataLoader, TensorDataset  # For batch processing
from src.utils import evaluate, extractEdgesFromMatrix  # Utility functions for evaluation and edge extraction
from src.Model import VAE_EAD  # Variational Autoencoder with Edge Attention Decoder
from scipy.sparse import csc_matrix  # Sparse matrix representation
from src.utils import normalize_sparse_hypergraph_symmetric  # Hypergraph normalization
import codecs
import csv

# Specify tensor type for GPU computations
Tensor = torch.cuda.FloatTensor


# Utility function to save data to a CSV file
def data_write_csv(file_name, datas):
    """
    Save data to a CSV file.
    Args:
        file_name (str): Path to save the CSV file.
        datas (list): Data to be written to the file.
    """
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # Open file in write mode
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)  # Write each row to the CSV file
    print("Save Successfully")


# Define the GRN inference model class for cell-type-specific GRN
class celltype_GRN_model:
    def __init__(self, opt):
        """
        Initialize the model with configuration options.
        Args:
            opt: Configuration options (e.g., learning rate, dataset paths).
        """
        self.opt = opt
        # Create a directory for saving results
        try:
            os.mkdir(opt.save_name)
        except FileExistsError:
            print('Directory already exists')

    def initalize_A(self, data):
        """
        Initialize the adjacency matrix A with random values and remove self-loops.
        Args:
            data: Input data (gene expression matrix).
        Returns:
            A: Initialized adjacency matrix.
        """
        num_genes = data.shape[1]  # Number of genes
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + \
            (np.random.rand(num_genes * num_genes) * 0.0002).reshape([num_genes, num_genes])
        np.fill_diagonal(A, 0)  # Remove self-loops by setting diagonal elements to 0
        return A

    def init_data(self):
        """
        Load and preprocess scRNA-seq data and ground truth for GRN inference.
        Returns:
            Processed data and evaluation masks.
        """
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)  # Ground truth GRN
        data = sc.read(self.opt.data_file)  # Load scRNA-seq data
        data = data.transpose()  # Transpose to align cells and genes
        gene_name = list(data.var_names)  # Extract gene names
        data_values = data.X

        # Create binary adjacency matrix
        adj = data_values.copy()
        adj[adj > 0] = 1
        adj = torch.tensor(adj)

        # Create dropout mask for missing values
        Dropout_Mask = (data_values != 0).astype(float)

        # Normalize data by removing batch effects
        means, stds = [], []
        for i in range(data_values.shape[1]):
            tmp = data_values[:, i]
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())
        means, stds = np.array(means), np.array(stds)
        stds[np.isnan(stds) | np.isinf(stds)] = 0
        data_values = (data_values - means) / stds
        data_values[np.isnan(data_values) | np.isinf(data_values)] = 0
        data_values = np.clip(data_values, -10, 10)

        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)

        # Create evaluation and TF masks
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        TF_mask = np.zeros([num_genes, num_genes])

        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i != j and item2 in TF:
                    Evaluate_Mask[i, j] = 1 if item in All_gene else 0
                    TF_mask[i, j] = 1

        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask), torch.FloatTensor(adj))
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)

        # Construct ground truth adjacency matrix
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges = set(zip(idx_send, idx_rec))

        return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name, adj

    def train_model(self):
        """
        Train the GRN inference model using a VAE-based architecture.
        """
        dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name, adj = self.init_data()
        adj_A_init = self.initalize_A(data)  # Initialize adjacency matrix
        vae = VAE_EAD(adj_A_init, self.opt.batch_size, 1, self.opt.n_hidden, self.opt.K, 
                      self.opt.dropout, self.opt.heads, 0.2).float().cuda()

        optimizer = optim.RMSprop(vae.parameters(), lr=self.opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=self.opt.lr * 0.2)  # Optimizer for adjacency matrix
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_step_size, gamma=self.opt.gamma)

        best_Epr, patience, counter = 0, 20, 0
        for epoch in range(self.opt.n_epochs + 1):
            # Training steps
            # Adjust learning phase for adjacency matrix updates
            vae.adj_A.requires_grad = epoch % (self.opt.K1 + self.opt.K2) >= self.opt.K1

            for data_batch in dataloader:
                inputs, _, dropout_mask, adj = data_batch
                inputs = Variable(inputs.type(Tensor))
                adj = normalize_sparse_hypergraph_symmetric(csc_matrix(adj.t()))

                # Forward pass and compute loss
                latent_z, loss_rec, loss_kl, loss_kl2, _ = vae(inputs, adj, dropout_mask=dropout_mask.cuda(), 
                                                               temperature=max(0.95 ** epoch, 0.5), opt=self.opt)
                sparse_loss = torch.mean(torch.abs(vae.adj_A))
                lossback = loss_rec + self.opt.beta * loss_kl + self.opt.omega * loss_kl2 + self.opt.alpha * sparse_loss

                # Backpropagation
                lossback.backward()
                if vae.adj_A.requires_grad:
                    optimizer2.step()
                else:
                    optimizer.step()

            scheduler.step()

            # Evaluation phase
            if vae.adj_A.requires_grad:
                Ep, Epr = evaluate(vae.adj_A.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
                best_Epr = max(Epr, best_Epr)

        # Save results
        extractEdgesFromMatrix(vae.adj_A.cpu().detach().numpy(), gene_name, TFmask2).to_csv(
            f"{self.opt.save_name}/GRN_inference.tsv", sep='\t', index=False)
