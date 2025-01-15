# Import required libraries
import os
import numpy as np
import pandas as pd
import scanpy as sc  # For single-cell RNA sequencing data processing
import torch
import torch.optim as optim  # Optimizers for training
from torch.autograd import Variable  # For handling tensor operations
from torch.utils.data import DataLoader, TensorDataset  # For batch data handling
from src.Model import VAE_EAD  # Import the Variational Autoencoder with Edge Attention Decoder
from src.utils import evaluate, extractEdgesFromMatrix  # Utility functions for evaluation and edge extraction
from scipy.sparse import csc_matrix  # For sparse matrix representation
from src.utils import normalize_sparse_hypergraph_symmetric  # For hypergraph normalization
import codecs  # For handling CSV encoding
import csv  # For CSV operations
import time  # For tracking runtime
import psutil  # For monitoring system performance

# Specify tensor type for GPU
Tensor = torch.cuda.FloatTensor

# Utility function to write data to a CSV file
def data_write_csv(file_name, datas):
    """
    Save data to a CSV file.
    Args:
        file_name (str): Path to the CSV file.
        datas (list): Data to be saved.
    """
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # Open the file in write mode
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)  # Write each row to the file
    print("Save Successfully")

# Define the model class for non-cell-type-specific GRN inference
class non_celltype_GRN_model:
    def __init__(self, opt):
        """
        Initialize the GRN model with the provided options.
        Args:
            opt: Configuration options for the model.
        """
        self.opt = opt
        try:
            os.mkdir(opt.dataset_name)  # Create a directory for dataset-specific outputs
        except:
            print('Directory already exists')

    # Initialize the adjacency matrix A with random values
    def initalize_A(self, data):
        """
        Initialize the adjacency matrix with small random values and no self-loops.
        Args:
            data: Input gene expression data.
        Returns:
            A: Initialized adjacency matrix.
        """
        num_genes = data.shape[1]  # Number of genes
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + \
            (np.random.rand(num_genes * num_genes) * 0.0002).reshape([num_genes, num_genes])
        np.fill_diagonal(A, 0)  # Set diagonal elements to 0 (no self-loops)
        return A

    # Load and preprocess data
    def init_data(self):
        """
        Load, preprocess scRNA-seq data, and prepare training/evaluation masks.
        Returns:
            Preprocessed data and masks for training and evaluation.
        """
        # Load ground truth GRN and scRNA-seq data
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)  # Gene interaction ground truth
        data = sc.read(self.opt.data_file)  # Read scRNA-seq data
        data = data.transpose()  # Transpose to align cells and genes

        # Normalize and prepare data
        gene_name = list(data.var_names)  # List of gene names
        data_values = data.X
        adj = data_values.copy()  # Create adjacency matrix
        adj[adj > 0] = 1  # Binarize the adjacency matrix
        adj = torch.tensor(adj)
        Dropout_Mask = (data_values != 0).astype(float)  # Create dropout mask
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0))  # Normalize expression data
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)

        # Create evaluation and TF masks
        TF = set(Ground_Truth['Gene1'])  # Set of transcription factors (TFs)
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])  # All genes in ground truth
        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask[i, j] = 1  # Mark edges for evaluation
                if item2 in TF:
                    TF_mask[i, j] = 1  # Mark TF-related edges

        # Prepare training data
        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask), torch.FloatTensor(adj))
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=8)

        # Prepare ground truth adjacency matrix
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges = set(zip(idx_send, idx_rec))

        return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name, adj

    # Train the GRN inference model
    def train_model(self):
        """
        Train the GRN inference model using a VAE architecture.
        """
        # Initialize data and model components
        dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name, adj = self.init_data()
        adj_A_init = self.initalize_A(data)  # Initialize adjacency matrix
        vae = VAE_EAD(adj_A_init, self.opt.batch_size, 1, self.opt.n_hidden, self.opt.K, self.opt.dropout, 
                      self.opt.heads, 0.2).float().cuda()

        # Optimizers and schedulers
        optimizer = optim.RMSprop(vae.parameters(), lr=self.opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=self.opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_step_size, gamma=self.opt.gamma)

        # Training loop
        best_Epr, patience, counter = 0, 20, 0
        for epoch in range(self.opt.n_epochs + 1):
            # Update adjacency matrix and optimize VAE
            if epoch % (self.opt.K1 + self.opt.K2) < self.opt.K1:
                vae.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
            # Train the model using batches
            for data_batch in dataloader:
                optimizer.zero_grad()
                # Data processing and loss computation
                inputs, _, dropout_mask, adj = data_batch
                inputs = Variable(inputs.type(Tensor))
                adj = normalize_sparse_hypergraph_symmetric(csc_matrix(adj.t()))
                latent_z, loss_rec, loss_kl, loss_kl2, _ = vae(inputs, adj, dropout_mask=None,
                                                                temperature=max(0.95 ** epoch, 0.5), opt=self.opt)
                sparse_loss = torch.mean(torch.abs(vae.adj_A))
                loss = loss_rec + self.opt.beta * loss_kl + self.opt.omega * loss_kl2 + self.opt.alpha * sparse_loss
                loss.backward()
                if epoch % (self.opt.K1 + self.opt.K2) < self.opt.K1:
                    optimizer.step()
                else:
                    optimizer2.step()
            scheduler.step()

            # Evaluate the model
            if epoch % (self.opt.K1 + self.opt.K2) >= self.opt.K1:
                Ep, Epr = evaluate(vae.adj_A.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
                if Epr >= best_Epr:
                    best_Epr, counter = Epr, 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping due to no improvement")
                        break

        # Save results
        extractEdgesFromMatrix(vae.adj_A.cpu().detach().numpy(), gene_name, TFmask2).to_csv(
            f"{self.opt.dataset_name}/GRN_inference_link.tsv", sep='\t', index=False)
        data_write_csv(f"{self.opt.dataset_name}/GRN_inference_result.csv", lossData)
