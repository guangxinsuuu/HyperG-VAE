# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc  # Library for handling single-cell data
import torch.optim as optim  # Optimizers for training
from torch.autograd import Variable  # For automatic differentiation
from torch.utils.data import DataLoader, TensorDataset  # To load data in batches
from src.utils import evaluate, extractEdgesFromMatrix  # Utility functions for evaluation and edge extraction
from src.Model import VAE_EAD  # Variational Autoencoder with Edge Attention Decoder
import time
import psutil
import codecs
import csv
from scipy.sparse import csc_matrix  # Sparse matrix representation
from src.utils import normalize_sparse_hypergraph_symmetric  # For hypergraph normalization

# Specify tensor type for GPU
Tensor = torch.cuda.FloatTensor

# Utility function for saving data to a CSV file
def data_write_csv(file_name, datas):
    """
    Write data to a CSV file.
    Args:
        file_name (str): Path to save the CSV file.
        datas (list): List of data rows to write.
    """
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # Open file in write mode
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)  # Write each row
    print("Save Successfully")


# Class definition for testing a non-cell-type-specific GRN model
class test_non_celltype_GRN_model:
    def __init__(self, opt):
        """
        Initialize the GRN model with configuration options.
        Args:
            opt: A configuration object containing options like dataset path, learning rate, etc.
        """
        self.opt = opt
        try:
            os.mkdir(opt.save_name)  # Create directory for saving results
        except:
            print('Directory already exists')

    def initalize_A(self, data):
        """
        Initialize the adjacency matrix (A) with random values, ensuring no self-loops.
        Args:
            data: Input data (gene expression matrix).
        Returns:
            A: Initialized adjacency matrix.
        """
        num_genes = data.shape[1]  # Number of genes
        # Initialize adjacency matrix with small random values and uniform distribution
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + \
            (np.random.rand(num_genes * num_genes) * 0.0002).reshape([num_genes, num_genes])
        np.fill_diagonal(A, 0)  # Remove self-loops by setting diagonal elements to 0
        return A

    def init_data(self):
        """
        Load and preprocess scRNA-seq data, and prepare it for training.
        Returns:
            Data loader, number of nodes, number of genes, data frame, gene names, and adjacency matrix.
        """
        data = sc.read(self.opt.data_file)  # Load scRNA-seq data
        
        # Extract gene names and data matrix
        gene_name = list(data.var_names)  # Gene names from scRNA-seq data
        data_values = data.X  # Gene expression values

        print('Data shape: ' + str(data_values.shape))  # Log the data shape

        # Create adjacency matrix (binarized for hypergraph structure)
        adj = data.X.copy()
        adj[adj > 0] = 1  # Binarize adjacency matrix
        adj = torch.tensor(adj)

        # Dropout mask for missing values
        Dropout_Mask = (data_values != 0).astype(float)

        # Convert data to DataFrame for easier handling
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)

        # Number of genes and nodes
        num_genes, num_nodes = data.shape[1], data.shape[0]

        # PyTorch DataLoader for training
        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask), torch.FloatTensor(adj))
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)
        print(num_nodes)  # Log the number of nodes
        return dataloader, num_nodes, num_genes, data, gene_name, adj

    def train_model(self):
        """
        Train the VAE-based GRN inference model.
        """
        # Load data and initialize adjacency matrix
        opt = self.opt
        dataloader, num_nodes, num_genes, data, gene_name, adj = self.init_data()
        adj_A_init = self.initalize_A(data)  # Initialize adjacency matrix

        # Initialize the VAE model
        vae = VAE_EAD(adj_A_init, opt.batch_size, 1, opt.n_hidden, opt.K, opt.dropout, opt.heads, 0.2).float().cuda()

        # Define optimizers
        optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)

        # Initialize variables for early stopping
        best_loss = 1e6  # Track the best loss
        patience = 20  # Early stopping patience
        lossData = [[]]
        counter = 0  # Counter for early stopping
        start_time = time.time()  # Start tracking time

        # Training loop
        for epoch in range(opt.n_epochs + 1):
            data_embedding = torch.tensor([]).cuda()  # Placeholder for latent embeddings
            loss_all, mse_rec, loss_klmlp, loss_klgcn, loss_sparse = [], [], [], [], []

            # Adjust adjacency matrix training phase
            vae.adj_A.requires_grad = epoch % (opt.K1 + opt.K2) >= opt.K1

            for i, data_batch in enumerate(dataloader):
                if len(data_batch[0]) == opt.batch_size:
                    optimizer.zero_grad()  # Reset gradients
                    inputs, data_id, dropout_mask, adj = data_batch
                    inputs = Variable(inputs.type(Tensor))
                    adj = normalize_sparse_hypergraph_symmetric(csc_matrix(adj.t()))

                    # Forward pass through the VAE
                    latent_z, loss_rec, loss_kl, loss_kl2, latent_2 = vae(inputs, adj, dropout_mask=None,
                                                                          temperature=max(0.95 ** epoch, 0.5), opt=opt)

                    # Accumulate embeddings for latent space visualization
                    if epoch > 0:
                        data_embedding = torch.cat((data_embedding, latent_z), dim=0)

                    # Compute sparse loss
                    sparse_loss = torch.mean(torch.abs(vae.adj_A))

                    # Total loss
                    lossback = loss_rec + loss_kl * opt.beta + loss_kl2 * opt.omega + opt.alpha * sparse_loss
                    lossback.backward()  # Backpropagation

                    # Collect loss values
                    mse_rec.append(loss_rec.item())
                    loss_all.append(lossback.item())
                    loss_klmlp.append(loss_kl.item())
                    loss_klgcn.append(loss_kl2.item())
                    loss_sparse.append(sparse_loss.item())

                    # Update parameters
                    if vae.adj_A.requires_grad:
                        optimizer2.step()
                    else:
                        optimizer.step()

            scheduler.step()  # Adjust learning rate

            # Early stopping and logging
            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                avg_loss = np.mean(loss_all)
                if avg_loss <= best_loss:
                    best_loss = avg_loss
                    counter = 0
                else:
                    counter += 1
                if counter >= patience:
                    print("Early stopping due to no improvement")
                    break

                print(f"Epoch: {epoch}, Loss: {avg_loss}, MSE: {np.mean(mse_rec)}, KL_MLP: {np.mean(loss_klmlp)}, "
                      f"KL_GCN: {np.mean(loss_klgcn)}, Sparse Loss: {np.mean(loss_sparse)}")

        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")

        # Save results
        latent_z = pd.DataFrame(data_embedding.cpu().detach().numpy())
 

        latent_z.to_csv("zeisel2\GRN_latent1_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv")

        latent_2 = pd.DataFrame(latent_2.cpu().detach().numpy())
        latent_2.to_csv("zeisel2\GRN_latent2_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv")

        data_write_csv("zeisel2\Result_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv", lossData)



