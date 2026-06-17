import sys
import torch
from torch import autograd, optim, nn
import numpy as np
import fewshot_re_kit
from torch.distributions import MultivariateNormal



class Proto(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, dot=False, k=50, alpha=0.21, num_pseudo_samples=50):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.dot = dot
        self.hidden_size = 768
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.k = k
        self.alpha_dc = alpha
        self.num_pseudo_samples = num_pseudo_samples
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def __euclid_dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_euclid_dist__(self, S, Q):
        return self.__euclid_dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)


    def distribution_calibration(self, support, query, k, alpha=None):
        B, N, K, D = support.size()
        _, total_Q, _ = query.size()

        support_ = support.view(B * N * K, D)
        query_ = query.view(B * total_Q, D)
        dist_matrix = torch.cdist(support_, query_)

        effective_k = min(k, dist_matrix.size(1))
        topk_dist, indices = torch.topk(dist_matrix, k=effective_k, dim=1, largest=False)

        nearest_query_samples = torch.gather(
            query_.unsqueeze(0).expand(B * N * K, -1, -1),
            1,
            indices.unsqueeze(-1).expand(-1, -1, D)
        )

        query_nearest_mean = torch.mean(nearest_query_samples, dim=1)  # (B*N*K, D)

        calibrated_means = (support_ + query_nearest_mean) / 2
        query_vars = torch.var(nearest_query_samples, dim=1, unbiased=False)  # (B*N*K, D)
        global_vars = torch.var(support_, dim=0, unbiased=False)  # (D,)
        calibrated_vars = global_vars + (alpha if alpha is not None else 0.0) * query_vars  # (B*N*K, D)
        calibrated_covs = torch.stack([torch.diag(v) for v in calibrated_vars], dim=0)  # (B*N*K, D, D)

        calibrated_means = calibrated_means.view(B, N, K, D)
        calibrated_covs = calibrated_covs.view(B, N, K, D, D)

        return calibrated_means, calibrated_covs

    def generate_pseudo_samples(self, means, covs, num_samples):
        B, N, K, D = means.size()
        pseudo_samples = []
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    mean = means[b, n, k]
                    cov = covs[b, n, k]
                    dist = MultivariateNormal(mean, covariance_matrix=cov)
                    samples = dist.sample((num_samples,))
                    pseudo_samples.append(samples)

        pseudo_samples = torch.stack(pseudo_samples, dim=0).view(B, N, -1, D)
        return pseudo_samples


    def forward(self, support, query, rel_txt, N, K, total_Q):
        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc, 1)  # [B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1)
        support_h, support_t, s_loc = self.sentence_encoder(support)
        query_h, query_t, q_loc = self.sentence_encoder(query)
        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)
        support = support.view(-1, N, K, self.hidden_size * 2)
        query = query.view(-1, total_Q, self.hidden_size * 2)
        calibrated_means, calibrated_covs = self.distribution_calibration(support, query, self.k, self.alpha_dc)
        pseudo_samples = self.generate_pseudo_samples(calibrated_means, calibrated_covs, self.num_pseudo_samples)

        support = torch.cat([support, pseudo_samples], dim=2)
        rel_rep = rel_rep.view(-1, N, rel_gol.shape[1] * 2)
        rel_rep = rel_rep.unsqueeze(2).expand(-1,-1,support.size(2),-1)
        support = support + rel_rep
        prototypes = torch.mean(support, dim=2)
        logits = self.__batch_dist__(prototypes, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        return logits, pred