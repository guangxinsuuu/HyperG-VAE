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
import time
import psutil
import codecs
import csv
from scipy.sparse import csc_matrix
from src.utils import normalize_sparse_hypergraph_symmetric
Tensor = torch.cuda.FloatTensor
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("Save Successfully")


class test_non_celltype_GRN_model:
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
        data = sc.read(self.opt.data_file)
        
        
        #data = data.transpose()
        gene_name = list(data.var_names)
        data_values = data.X


        print('data shape: '+str(data_values.shape))
        adj = data.X.copy()
        adj[adj>0]=1
        adj = torch.tensor(adj)
        Dropout_Mask = (data_values != 0).astype(float)
        #data_values = (data_values - data_values.mean(0)) / (data_values.std(0))#!!!处理完删掉备注：是不是需要删除这行
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)
        num_genes, num_nodes = data.shape[1], data.shape[0]
        feat_train = torch.FloatTensor(data.values)
        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask),torch.FloatTensor(adj))
        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)
        print(num_nodes)
        return dataloader,  num_nodes, num_genes, data, gene_name, adj


    def train_model(self):
        opt = self.opt
        dataloader,  num_nodes, num_genes, data, gene_name, adj = self.init_data()
        adj_A_init  = self.initalize_A(data)

        bactchsize = opt.batch_size
        vae = VAE_EAD(adj_A_init, bactchsize, 1, opt.n_hidden, opt.K, opt.dropout, opt.heads, 0.2).float().cuda()

        # vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K).float().cuda()
        optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
        best_loss = 1000000
        vae.train()
        print(vae)
        patience = 20

        lossData=[[]]
        counter = 0
        start_time = time.time()
        for epoch in range(opt.n_epochs+1):
            # print('epoch '+str(epoch))
            data_embedding = torch.tensor([]).cuda()
            loss_all, mse_rec, loss_klmlp, data_ids, loss_klgcn, loss_sparse = [], [], [], [], [], []
            if epoch % (opt.K1+opt.K2) < opt.K1:
                vae.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                #print('batch '+str(i))
                if len(data_batch[0]) == opt.batch_size:
                    optimizer.zero_grad()
                    inputs, data_id, dropout_mask, adj = data_batch
                    inputs = Variable(inputs.type(Tensor))
                    data_ids.append(data_id.cpu().detach().numpy())
                    temperature = max(0.95 ** epoch, 0.5)

                    adj = normalize_sparse_hypergraph_symmetric(csc_matrix(adj.t()))
                    #print('--------------------')
                    #print(data_ids)
                    latent_z, loss_rec, loss_kl, loss_kl2, latent_2 = vae(inputs, adj, dropout_mask=None,
                                                                            temperature=temperature, opt=opt)
                    #print(data_ids)
                    
                    # print("####")
                    # print(inputs)

                    
                    #for data embedding!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if epoch > 0:
                        data_embedding = torch.cat((data_embedding, latent_z), dim=0)

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
                    if epoch % (opt.K1 + opt.K2) < opt.K1:
                        optimizer.step()
                    else:
                        optimizer2.step()
                    
                    
            
            scheduler.step()
            # print("$$$")
            # print(inputs)


            ###############
            # data_ids = []
            # embeds = []
            # for i, data_batch in enumerate(dataloader, 0):
            #     if len(data_batch[0]) == opt.batch_size:
            #         optimizer.zero_grad()
            #         inputs, data_id, dropout_mask, adj = data_batch
            #         inputs = Variable(inputs.type(Tensor))
            #         adj = normalize_sparse_hypergraph_symmetric(csc_matrix(adj.t()))
            #         print('xxxxxxxxxxxxx')
            #         latent_z, loss_rec, loss_kl, loss_kl2, latent_2 = vae(inputs, adj, dropout_mask=None,
            #                                                                     temperature=temperature, opt=opt)
            #         data_ids.append(data_id.detach().numpy())
            #         embeds.append(latent_z.cpu().detach().numpy())
            #         print('ytyyyyyyyyyy')
            # data_ids = np.hstack(data_ids)
            # embeds = np.vstack(embeds)
            # data_id_map = np.zeros(len(data_ids))
            # for i, item in enumerate(data_ids):
            #     data_id_map[item] = i
            # embeds = embeds[data_id_map.astype(int)]
            # print('xxxxxxxxxxxxx')
            # print(embeds)
            # print('xxxxxxxxxxxxx')
            # adata = sc.AnnData(embeds)
            # adata.write_h5ad("zeisel\GRN_latent1_" + 'embedding.h5ad')


            ###############

            if epoch % (opt.K1+opt.K2) >= opt.K1:

                lossData.append([np.mean(loss_all)])
                if np.mean(loss_all) <= best_loss:
                    best_loss = np.mean(loss_all)
                    counter = 0
                else: 
                    counter+=1
                if counter >= patience:
                    print("No Patience")
                    break
                    # print("latent: "+str(latentvalue))
                
                if np.isnan(np.mean(loss_all)):
                        print("Loss is NaN. Training terminated.")
                        break
                
                print('epoch:', epoch,  'loss:',
                      np.mean(loss_all), 'mse_loss:', np.mean(mse_rec), 'kl_loss:', np.mean(loss_klmlp), 'klgcn_loss:', np.mean(loss_klgcn),'sparse_loss:',
                      np.mean(loss_sparse))
               


        end_time = time.time()

        print(f"Running time: {(end_time - start_time)/60} mins")

        

        np.savetxt("zeisel2\Data_ids_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv", np.concatenate(data_ids), delimiter=",", fmt="%d")
        extractEdgesFromMatrix(vae.adj_A.cpu().detach().numpy(), gene_name,None).to_csv(
        "zeisel2\TSV_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".tsv", sep='\t', index=False)
        
        latent_z = pd.DataFrame(data_embedding.cpu().detach().numpy())
        latent_z.to_csv("zeisel2\GRN_latent1_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv")

        latent_2 = pd.DataFrame(latent_2.cpu().detach().numpy())
        latent_2.to_csv("zeisel2\GRN_latent2_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv")

        data_write_csv("zeisel2\Result_"+str(opt.dataset_name)+str(opt.n_hidden)+str('_alpha_')+str(opt.nalpha)+str('_beta_')+str(opt.nbeta)+str('_omega_')+str(opt.nomega)+str('_lr_')+str(opt.nlr)+str('_wd_')+str(opt.nlr_step_size)+str('_seed_')+str(opt.seed)+str('_head_')+str(opt.heads)+str('_dropout_')+str(opt.dropoutrate)+str('_batch_')+str(opt.batch_size)+".csv", lossData)



