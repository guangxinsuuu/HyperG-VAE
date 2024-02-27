import numpy as np
import torch
import torch.nn.functional  as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn.parameter import Parameter
import math
Tensor = torch.cuda.FloatTensor

###########################
def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*torch.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5
##############################


def kl_loss(z_mean, z_var):
    mean_sq = torch.norm(z_mean)
    stddev_sq = z_var
    #return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
# We followed implement in https://github.com/jariasf/GMVAE/tree/master/pytorch
class LossFunctions:#1003: 用到hypergraph时，q(H|V,E)部分需要修改的！！！对应的新产生的hypergraph
    eps = 1e-8

    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type='mse'): #需要修改input 0313
        if rec_type == 'mse':
            if dropout_mask is None:
                loss = torch.mean((real - predicted).pow(2))
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(dropout_mask)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none').mean()
        else:
            raise Exception
        return loss

    def log_normal(self, x, mu, var):

        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        
        return loss.mean()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))




class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu.squeeze(2), logvar.squeeze(2)
    
class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1.double(), M2.double())

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g.double(), M2.t().double())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t().double(), g.double())

        return g1, g2

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
       
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #输出是x本身就行吧
        #return F.log_softmax(x, dim=1)
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)  
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, input_features, adj):  
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

class GCNLatent(nn.Module):
    def __init__(self, x_dim, z_dim, nonLinear):
        super(GCNLatent, self).__init__()
        
        self.latentnet = torch.nn.ModuleList([
            nn.Linear(x_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            Gaussian(z_dim, 1)
        ])

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)#和vae好像还是不一样的？还是技巧，哪个好
        z = mu + noise * std#为啥不是exp(std) 答：forward那块加了

        return z
    
    def latent(self, x):

        for layer in self.latentnet:
            x = layer(x)
        return x

    def forward(self, x, ):
       
        mu, logvar = self.latent(x.float())
        var = torch.exp(logvar)
        z = self.reparameterize(mu, var)
        output = {'lmean'  : mu, 'lvar': var, 'lvalue': z,}
        return output
    
class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(InferenceNet, self).__init__()
       
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            Gaussian(z_dim, 1)
        ])

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)#和vae好像还是不一样的？还是技巧，哪个好
        z = mu + noise * std#为啥不是exp(std) 答：forward那块加了

        return z

    def qyx(self, x, temperature):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def qzxy(self, x):

        for layer in self.inference_qzyx:
            x = layer(x)
        return x

    def forward(self, x, adj, temperature=1.0):
        
        mu, logvar = self.qzxy(x)

        mu = torch.matmul(mu, adj)
        logvar = torch.matmul(logvar, adj)
        var = torch.exp(logvar)
        z = self.reparameterize(mu, var)
        output = {'mean'  : mu, 'var': var, 'gaussian': z,
                 }
        return output


class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, n_gene, nonLinear):
        super(GenerativeNet, self).__init__()
        self.n_gene = n_gene
        self.y_mu = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))
        self.y_var = nn.Sequential(nn.Linear(y_dim, z_dim), nonLinear, nn.Linear(z_dim, n_gene))

        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(1, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_logvar = self.y_var(y)
        return y_mu, y_logvar

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, adj):
        
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        output = {'x_rec': x_rec}
      
        return output

class GCNGenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, nonLinear):
        super(GCNGenerativeNet, self).__init__()


        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(1, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, ):
        
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        output = x_rec
      
        return output
    
class VAE_EAD(nn.Module):
    def __init__(self, adj_A, batchsize, x_dim, z_dim, y_dim, dropout, heads, leakyslope):
        super(VAE_EAD, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True, name='adj_A'))
        self.n_gene = n_gene = len(adj_A)
        nonLinear = nn.Tanh()
        self.inference = InferenceNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim, n_gene, nonLinear)
        #==============================================#
        self.gcy1 = GraphConvolution(batchsize, z_dim)
        self.gcy2 = GraphConvolution(z_dim, n_gene)
        self.latent = GCNLatent(x_dim, z_dim, nonLinear)
        self.gcngenerative = GCNGenerativeNet(x_dim, z_dim, nonLinear)
        #==============================================#
        self.gat = GAT(nfeat=batchsize, 
                nhid=z_dim, 
                nclass=batchsize, 
                dropout=dropout, 
                nheads=heads, 
                alpha=leakyslope)
        #================================================#
        self.losses = LossFunctions()
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def _one_minus_A_t(self, adj):
        adj_normalized = Tensor(np.eye(adj.shape[0])) - (adj.transpose(0, 1))
        return adj_normalized

    def forward(self, x, adj, dropout_mask, temperature=1.0, opt=None, ):
        # print(x)
        #todo: 去掉gumbel-softmax看效果
        neg_slope = 0.2
        x_ori = x
        x = x.view(x.size(0), -1, 1)
        mask = Variable(torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(), requires_grad=False).cuda()
        adj_A_t = self._one_minus_A_t(self.adj_A * mask) #(I-A^T)
        adj_A_t_inv = torch.inverse(adj_A_t) #(I-A^T)^(-1)
        out_inf = self.inference(x, adj_A_t, temperature)
   
        
        
        z_inv = torch.matmul(out_inf['gaussian'], adj_A_t_inv)
        out_gen = self.generative(z_inv, adj_A_t)
        output = out_inf

        # -----------filtering on hyperedges------------#
        #todo: y_features是简单的转置，对角阵还是随机游走之类的学
        adj = torch.tensor(adj.todense()).cuda()


        ###########
        gatout = self.gat(x_ori.t(), adj)
        #############

        # hy = self.gcy1(x_ori.t(), adj)
        # hy = F.leaky_relu(hy, negative_slope=neg_slope)        
        
        # hy = self.gcy2(hy, adj)
        # hy = F.leaky_relu(hy, negative_slope=neg_slope)
        #hy_out = self.latent(hy.view(hy.size(0),-1,1))
        hy_out = self.latent(gatout.view(gatout.size(0),-1,1))
        
        lmean, lvar, lvalue = hy_out['lmean'], hy_out['lvar'], hy_out['lvalue']

        gcnout = self.gcngenerative(lvalue)
        # ----------------------------------------------#

        for key, value in out_gen.items():
            output[key] = value #output为decoder部分，z为latent space
        dec = output['x_rec']

        ############
        output['x_rec'] = output['x_rec'].float().mul(gcnout.t().float())
        ##############

        
        loss_rec = self.losses.reconstruction_loss(x_ori, output['x_rec'], dropout_mask, 'mse')*(0.5)/1
        loss = nll_gaussian( output['x_rec'],x_ori,torch.tensor(0))
      
        loss_kl = kl_loss(output['mean'], output['var'])
        loss_kl2 = kl_loss(lmean,lvar)
        return out_inf['mean'], loss_rec, loss_kl, loss_kl2, lmean

        