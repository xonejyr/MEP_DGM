import torch
import torch.nn as nn

# the realization of normalization flow model

class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        # forward transform of the data by block
        x = z
        for i in range(len(self.t)): # len(self.t) = 2
            x_ = x * self.mask[i] # [0, 1, 0, 1, 0, 1] for mask[0], [1, 0, 1, 0, 1, 0] for mask[1]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        # backward transform to recover the data
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1) # by math, for log (exp) obtained by differential of exp, if x ~ z.exp(s(z))+t, and - means that the x (actually mu^ - mu_g) is the minus of idealized (mu_g - mu^), z does not change, but jacobian is from + to -
        return z, log_det_J

    def log_prob(self, x):
        # calculate the log probability of the given data under the prior distribution
        # 1. x => z by backward_p
        # 2. log_prob(z) by prior
        # 3. add logp
        # final log probability is the sum of log_prob(z) and logp
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        # sample from the prior distribution (z) and then apply the forward transform, to get sample data x
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        # calculate the log probability of the given data under the prior distribution
        return self.log_prob(x)
