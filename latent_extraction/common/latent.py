import torch
from torch import nn
from torch.nn import LSTMCell
from torch.nn.functional import gumbel_softmax, logsigmoid
import numpy as np
from latent_extraction.nn.kuma_gate import KumaGate
from latent_extraction.nn.rcnn import RCNNCell
from torch.nn import Linear, Sequential, Dropout, Softplus, Tanh, ReLU
from torch.distributions import Bernoulli
from torch.distributions import kl_divergence
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

class IndependentLatentModel(nn.Module):
    """
    The latent model ("The Generator") takes an input text
    and returns samples from p(z|x)
    This version uses a reparameterizable distribution, e.g. HardKuma.
    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 2048,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 distribution: str = "kuma"
                 ):

        super(IndependentLatentModel, self).__init__()


        self.z_layer = KumaGate(768)

        self.z = None      # z samples
        self.z_dists = []  # z distribution(s)
        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x):

        h = x
        z_dist = self.z_layer(h)
        # we sample once since the state was already repeated num_samples
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]

        # mask invalid positions
        z = z.squeeze(-1)
        self.z = z  # [B, T]
        self.z_dists = [z_dist]

        return z

class DependentLatentModel(nn.Module):
    """
    The latent model ("The Generator") takes an input text
    and returns samples from p(z|x)
    This version uses a reparameterizable distribution, e.g. HardKuma.
    """

    def __init__(self,
                 hidden_size: int = 2*768,
                 z_rnn_size:  int = 30,
                 ):

        super(DependentLatentModel, self).__init__()

        enc_size = hidden_size 

        #self.z_cell = RCNNCell(enc_size + 1, z_rnn_size)
        self.z_cell = LSTMCell(enc_size + 1, z_rnn_size)

        self.z_cell.cuda()
        self.z_layer = KumaGate(enc_size + z_rnn_size)

        self.z_layer.cuda()
        self.z = None      # z samples
        self.z_dists = []  # z distribution(s)

        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask = None):

        # encode sentence
        batch_size, time, dim = x.shape
        h = x.transpose(0, 1)  # time, batch, dim

        z = []
        z_dists = []

        # initial states  [1, B, z_rnn_dim]
        if isinstance(self.z_cell, LSTMCell):  # LSTM
            state = h.new_zeros(
                [2 * batch_size, self.z_cell.hidden_size]).chunk(2)
        else:  # RCNN
            state = h.new_zeros([3 * batch_size, self.z_cell.hidden_size]).chunk(3)

        for h_t, t in zip(h, range(time)):

            # compute Binomial z distribution for this time step
            z_t_dist = self.z_layer(torch.cat([h_t, state[0]], dim=-1))
            z_dists.append(z_t_dist)

            # we sample once since the state was already repeated num_samples
            if self.training:
                z_t = z_t_dist.sample()  # [B, 1]
            else:
                # deterministic strategy
                p0 = z_t_dist.pdf(h.new_zeros(()))
                p1 = z_t_dist.pdf(h.new_ones(()))
                pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
                zero_one = torch.where(
                    p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
                z_t = torch.where((pc > p0) & (pc > p1),
                                  z_t_dist.mean(), zero_one)  # [B, M]

            z.append(z_t)

            # update cell state (to make dependent decisions)
            rnn_input = torch.cat([h_t, z_t], dim=-1)  # [B, 2D+1]
            state = self.z_cell(rnn_input, state)

        z = torch.stack(z, dim=1).squeeze(-1)  # [B, T]

        self.z = z
        self.z_dists = z_dists

        return z

class GumbelSoftmaxModel(nn.Module):
    def __init__(self,
                  in_features, out_features=1):
                 
        super(GumbelSoftmaxModel, self).__init__()
        self.z_layer = Sequential(
            Linear(in_features, out_features, bias=True),
            Softplus()
        ) 
        self.threshold = 0.5
        self.beta = 0.1

    def prior(self, var_size, device, threshold=0.5):
        p = torch.tensor([threshold], device=device)
        p = p.view(1, 1)
        p_prior = p.expand(var_size)  # [batch-size, k, feature dim]

        return p_prior

    def reparameterize(self, p_i, tau, k, num_sample=1):
        p_i_ = p_i.view(p_i.size(0), 1, 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, k, p_i_.size(-1))
        C_dist = RelaxedOneHotCategorical(tau, logits=p_i_)
        V = torch.max(C_dist.sample(), -2)[0]  # [batch-size, multi-shot, d]
        return V

    def forward(self, x, mask=None):
        
        batch_size = x.shape[0]
        single_output = self.z_layer(x)
        logits = single_output
        #logits = gumbel_softmax(single_output, hard=False, tau=0.01)

        
        p_i_prior = self.prior(var_size=logits.size(), device=logits.device, threshold=self.threshold)
        q_z = Bernoulli(logits=logits)
        p_z = Bernoulli(probs=p_i_prior)
        info_loss = (torch.distributions.kl_divergence(q_z, p_z).sum()) / batch_size
        info_loss = self.beta * info_loss
        return info_loss, logits

        