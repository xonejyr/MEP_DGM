import torch
import torch.nn as nn
import torch.distributions as distributions
from easydict import EasyDict

"""
通过神经网络来获得K
"""
# Assume existing modules
from .components.NFDP_parts.FPN_neck import FPN_neck_hm, FPNHead
from .components.NFDP_parts.Resnet import ResNet
from HeatmapBasis.utils import Softmax_Integral
from HeatmapBasis.builder import MODEL


class DynamicKBasis(nn.Module):
    def __init__(self, num_joints, max_bases=3, dim=2, ce_lambda=0.01, hid_dim=64):
        super(DynamicKBasis, self).__init__()
        self.num_joints = num_joints
        self.max_bases = max_bases
        self.dim = dim
        self.ce_lambda = ce_lambda  # Cross-entropy loss coefficient

        # Network to predict K_j (classification)
        self.k_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hid_dim), nn.ReLU(),
                nn.Linear(hid_dim, max_bases)  # Logits for K_j = 1, 2, ..., max_bases
            ) for _ in range(num_joints)
        ])

        # Weight networks
        self.weight_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hid_dim), nn.ReLU(),
                nn.Linear(hid_dim, max_bases), nn.Softmax(dim=-1)
            ) for _ in range(num_joints)
        ])

        # Basis networks
        self.basis_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hid_dim*2), nn.ReLU(),
                nn.Linear(hid_dim*2, max_bases * dim * 2)
            ) for _ in range(num_joints)
        ])

    def forward(self, pred_pts, gt_k=None):
        B, N, D = pred_pts.shape
        assert N == self.num_joints, f"Expected {self.num_joints} joints, got {N}"

        w = torch.zeros(B, N, self.max_bases, device=pred_pts.device)
        k_js = torch.zeros(B, N, dtype=torch.long, device=pred_pts.device)  # Store K_j for each batch and joint
        basis_dists = []
        ce_loss = 0.0

        for j in range(self.num_joints):
            pts_j = pred_pts[:, j, :]  # [B, D]
            
            # Predict K_j probabilities
            k_logits = self.k_nets[j](pts_j)  # [B, max_bases]
            k_probs = torch.softmax(k_logits, dim=-1)
            
            # Sample K_j (or use argmax during inference)
            if self.training:
                k_dist = distributions.Categorical(probs=k_probs)
                k_j = k_dist.sample() + 1  # [B], K_j in {1, ..., max_bases}
            else:
                k_j = k_probs.argmax(dim=-1) + 1  # [B]
            k_js[:, j] = k_j  # Store K_j

            # Compute cross-entropy loss if ground truth K_j is provided
            if self.training and gt_k is not None:
                gt_k_j = gt_k[:, j]  # [B]
                ce_loss += nn.functional.cross_entropy(k_logits, gt_k_j - 1)

            # Generate weights
            w_j = self.weight_nets[j](pts_j)  # [B, max_bases]
            w[:, j, :] = w_j

            # Generate Gaussian parameters
            params_j = self.basis_nets[j](pts_j).view(B, self.max_bases, 2 * D)
            mu_j = params_j[..., :D]
            log_sigma_j = params_j[..., D:]
            sigma_j = torch.exp(log_sigma_j)

            # Select top K_j components
            basis_dists_j = []
            for b in range(B):
                k_j_b = k_j[b].item() # 1
                w_j_b = w_j[b] # [max_bases]
                top_k_indices = torch.argsort(w_j_b, descending=True)[:k_j_b] # # k_j_b, 是w对应的编号
                for k in top_k_indices:
                    dist = distributions.Normal(mu_j[b, k], sigma_j[b, k]) 
                    basis_dists_j.append(dist) # 排在首位的是w最大值对应的分布，依次，直至攻击k_j_b个
            basis_dists.append(basis_dists_j) # 所有标志点

        total_loss = ce_loss * self.ce_lambda if self.training else 0.0
        return w, basis_dists, total_loss, k_js  # Return k_js

@MODEL.register_module
class HeatmapBasisNFR_numJoints_Dynamic_MLP_align(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(HeatmapBasisNFR_numJoints_Dynamic_MLP_align, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.fc_dim = cfg['NUM_FC_FILTERS']
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]
        self.hm_width_dim = self._preset_cfg['HEATMAP_SIZE'][1]
        self.hm_height_dim = self._preset_cfg['HEATMAP_SIZE'][0]
        self.max_bases = cfg['NUM_BASES']
        self.ce_lambda = cfg['KL_WEIGHT']
        self.hid_dim = cfg['HID_DIM']

        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        self.feature_channel = {18: 512, 34: 512, 50: 2048, 101: 2048, 152: 2048}[cfg['NUM_LAYERS']]
        self.decoder_feature_channel = { 
            18: [64, 128, 256, 512], 34: [64, 128, 256, 512],
            50: [256, 512, 1024, 2048], 101: [256, 512, 1024, 2048], 152: [256, 512, 1024, 2048]
        }[cfg['NUM_LAYERS']]

        self.fcs, out_channel = self._make_fc_layer()
        self.neck = FPN_neck_hm(
            in_channels=self.decoder_feature_channel,
            out_channels=self.decoder_feature_channel[0],
            num_outs=4,
        )
        self.head = FPNHead(
            feature_strides=(4, 8, 16, 32),
            in_channels=[self.decoder_feature_channel[0]] * 4,
            channels=128,
            num_classes=self.num_joints,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.integral_hm = Softmax_Integral(
            num_pts=self.num_joints,
            hm_width=self.hm_width_dim,
            hm_height=self.hm_height_dim
        )

        self.basis_dist = DynamicKBasis(num_joints=self.num_joints, max_bases=self.max_bases, ce_lambda=self.ce_lambda, hid_dim=self.hid_dim)
        self.fc_sigma = nn.Linear(self.feature_channel, self.num_joints * 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        import torchvision.models as tm
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained={cfg['PRETRAINED_RIGHT']})")

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in model_state and v.size() == model_state[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())
        return nn.Sequential(*fc_layers), input_channel

    def forward(self, x, labels=None):
        B = x.shape[0]
        feats = self.preact.forward_feat(x)
        feats_for_sigma = self.avg_pool(feats[-1]).reshape(B, -1)
        feats = self.neck(feats)
        output_hm = self.head(feats)

        out_coord = self.integral_hm(output_hm)
        pred_pts = out_coord.reshape(B, self.num_joints, 2)

        gt_k = labels['target_k'] if labels is not None and 'target_k' in labels else None
        w, basis_dists, k_loss, k_js = self.basis_dist(pred_pts, gt_k)  # Receive k_js
        self.basis_weights = w
        self.basis_dists = basis_dists

        out_sigma = self.fc_sigma(feats_for_sigma).reshape(B, self.num_joints, 2).sigmoid()
        scores = 1 - out_sigma.mean(dim=2, keepdim=True)

        basis_loss = None
        if self.training and labels is not None:
            gt_uv = labels['target_uv'].reshape(pred_pts.shape)
            bar_mu = (pred_pts - gt_uv) / out_sigma
            log_probs = torch.zeros(B, self.num_joints, self.max_bases, device=pred_pts.device)
            for j in range(self.num_joints):
                dists_j = basis_dists[j]
                for b in range(B):
                    k_j_b = k_js[b, j].item()  # Number of valid distributions for batch b, joint j
                    for idx, dist in enumerate(dists_j[:k_j_b]):  # Only process valid distributions
                        log_probs[b, j, idx] = dist.log_prob(bar_mu[b, j]).sum(dim=-1)
            weighted_log_probs = log_probs + torch.log(w + 1e-9)
            log_p = torch.logsumexp(weighted_log_probs, dim=2)
            basis_loss = -log_p.unsqueeze(2) + k_loss + torch.log(out_sigma)

            

            #lambda_edge = 0.1  # Adjust this coefficient as needed
            #basis_loss = basis_loss.mean() + lambda_edge * edge_loss

        output = EasyDict(
            pred_pts=pred_pts,
            heatmap=output_hm,
            sigma=out_sigma,
            maxvals=scores.float(),
            basis_weights=w,
            basis_dists=basis_dists,
            nf_loss=basis_loss
        )
        return output
    
    def _initialize(self):
        pass