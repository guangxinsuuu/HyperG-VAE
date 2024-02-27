import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from src.utils import evaluate, extractEdgesFromMatrix
from src.Model import VAE_EAD
from src.utils import evaluate, extractEdgesFromMatrix
from scipy.sparse import csc_matrix
from src.utils import normalize_sparse_hypergraph_symmetric
import codecs
import csv
Tensor = torch.cuda.FloatTensor


Tensor = torch.cuda.FloatTensor

def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("Save Successfully")

class celltype_GRN_model:
    def __init__(self,opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')
    def initalize_A(self,data):
        num_genes = data.shape[1]
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            [num_genes, num_genes])
        for i in range(len(A)):
            A[i, i] = 0
        return A


    def init_data(self,):
        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data = sc.read(self.opt.data_file)
        data = data.transpose()
        gene_name = list(data.var_names)
        data_values = data.X
        adj = data.X.copy()
        adj[adj>0]=1
        adj = torch.tensor(adj)
        Dropout_Mask = (data_values != 0).astype(float)
        means = []
        stds = []
        for i in range(data_values.shape[1]):
            tmp = data_values[:, i]
            means.append(tmp[tmp != 0].mean())
            stds.append(tmp[tmp != 0].std())
        means = np.array(means)
        stds = np.array(stds)
        stds[np.isnan(stds)] = 0
        stds[np.isinf(stds)] = 0
        data_values = (data_values - means) / (stds)
        data_values[np.isnan(data_values)] = 0
        data_values[np.isinf(data_values)] = 0
        data_values = np.maximum(data_values, -10)
        data_values = np.minimum(data_values, 10)
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask[i, j] = 1
                if item2 in TF:
                    TF_mask[i, j] = 1
        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask),torch.FloatTensor(adj))
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges = set(zip(idx_send, idx_rec))
        return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name,adj


    def train_model(self):
        dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name, adj  = self.init_data()
        opt = self.opt
        adj_A_init  = self.initalize_A(data)
        bactchsize = opt.batch_size
        vae = VAE_EAD(adj_A_init, bactchsize, 1,  self.opt.n_hidden, self.opt.K, opt.dropout, opt.heads, 0.2).float().cuda()
        Tensor = torch.cuda.FloatTensor
        optimizer = optim.RMSprop(vae.parameters(), lr=self.opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=self.opt.lr * 0.2)#这是什么操作
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_step_size, gamma=self.opt.gamma)
        best_Epr = 0
        vae.train()
        lossData=[[]]
        patience = 20
        counter = 0
        print(vae)
        for epoch in range(self.opt.n_epochs+1):
            loss_all, mse_rec, loss_klmlp, data_ids, loss_klgcn, loss_sparse = [], [], [], [], [], []
            if epoch % (self.opt.K1+self.opt.K2) < self.opt.K1:
                vae.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                if len(data_batch[0]) == opt.batch_size:
                    optimizer.zero_grad()
                    inputs, data_id, dropout_mask, adj = data_batch
                    inputs = Variable(inputs.type(Tensor))
                    data_ids.append(data_id.cpu().detach().numpy())
                    temperature = max(0.95 ** epoch, 0.5)#这是啥
                    adj = normalize_sparse_hypergraph_symmetric(csc_matrix(adj.t()))
                    latent_z, loss_rec, loss_kl, loss_kl2, latent_2  = vae(inputs, adj,dropout_mask=dropout_mask.cuda(),temperature=temperature,opt=self.opt)
                    #sparse loss和dag不一样
                    sparse_loss = torch.mean(torch.abs(vae.adj_A))
                        #sparse_loss = torch.mean(torch.abs(gcnadj))

                    lossback = loss_rec + loss_kl*opt.beta + loss_kl2*opt.omega + opt.alpha*sparse_loss

                    lossback.backward()

                    mse_rec.append(loss_rec.item())
                    loss_all.append(lossback.item())
                        
                    loss_klmlp.append(loss_kl.item())
                    loss_klgcn.append(loss_kl2.item())
                    loss_sparse.append(sparse_loss.item())

                    if epoch % (self.opt.K1+self.opt.K2) < self.opt.K1:
                        optimizer.step()
                    else:
                        optimizer2.step()
            scheduler.step()
            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                Ep, Epr,  = evaluate(vae.adj_A.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
                #Ep2, Epr2 = evaluate(gcnadj.cpu().detach().numpy(), truth_edges, Evaluate_Mask)
                lossData.append([Epr])
                if Epr >= best_Epr:
                    best_Epr = Epr
                    counter = 0
                else: 
                    counter+=1
                    if counter >= patience:
                        print("No Patience")
                        break
                    # print("latent: "+str(latentvalue))
                best_Epr = max(Epr, best_Epr)
                print('epoch:', epoch, 'Ep:', Ep, 'Epr:', Epr, 'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_klmlp), 'klgcn_loss:', np.mean(loss_klgcn),'sparse_loss:',
                      np.mean(loss_sparse))

        # data_embedding = pd.DataFrame(data_embedding.cpu().detach().numpy())
        # data_embedding.to_csv("tempbcell\Test_latent_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv")
   
        extractEdgesFromMatrix(vae.adj_A.cpu().detach().numpy(), gene_name, TFmask2).to_csv(
            opt.dataset_name + '/GRN_inference_link_'+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+".tsv", sep='\t', index=False)
        # pd.DataFrame(latentvalue.cpu().detach().numpy()).to_csv('temp\GRN_latent.csv')

        data_write_csv(opt.dataset_name+'/GRN_inference_result_'+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+".csv", lossData)
