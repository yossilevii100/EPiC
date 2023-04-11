"""
@Author: Meir Yossef Levi
@Contact: me.levi@campus.technion.ac.il
@File: epic_util
@Time: 23/04/10
"""

import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def get_knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    distances, indices = torch.sort(pairwise_distance, dim=-1, descending=True)

    k_indices = indices[:, :, :k]
    k_indices = k_indices.to(torch.int64)
    k_distances = distances[:, :, :k]
    return k_indices, k_distances


def extract_patches(pc, anchor, patch_size):
    # extract patch starting from anchor with a size of patch_size
    # input -
    # pc: Bx3xN
    # anchor: Bx1
    # patch_size: int
    # output -
    # patch: Bx3xK -> K is min(patch_size, N)

    B, num_dims, N = pc.shape

    # in case patch_size is larger than N, the entire point-cloud is considered as the patch
    actual_patch_size = min(patch_size, N)
    pc_neighbors, _ = get_knn(x=pc, k=actual_patch_size)

    anchor = anchor.unsqueeze(-1).repeat(1, 1, actual_patch_size)
    anchor_neighbors = torch.gather(pc_neighbors, 1, anchor).squeeze(1)

    indices_base = torch.arange(0, B, device=pc.device).view(-1, 1) * N  # Bx1

    anchor_neighbors = anchor_neighbors + indices_base
    anchor_neighbors = anchor_neighbors.view(-1)

    # Fetch the patch indices from the point cloud
    patch = pc.transpose(2, 1).contiguous()  # (B,N,num_dims)->(B*N,num_dims) :B*N*k+range(0,B*N)
    patch = patch.view(B * N, -1)[anchor_neighbors, :]
    patch = patch.view(B, actual_patch_size, num_dims)
    patch = patch.transpose(1, 2)  # Bx3XK

    return patch


def extract_random(pc: torch.Tensor, k: int) -> torch.Tensor:
    """Extracts K random points from point-cloud
    Parameters:
    pc (torch.Tensor): (B,3,N) point cloud
    k (int): number of random points to be fetched

    Returns:
    random_ppc (torch.Tensor): (B,3,K) random fetched point-cloud

   """
    # choose K random indices
    indices = torch.randperm(pc.shape[-1])
    random_indices = indices[:k]

    # Fetch the random indices from the point cloud
    random_ppc = pc[:, :, random_indices]
    return random_ppc


def extract_curves(pc, anchor, m, curve_size):
    # extract curve starting from anchor with a size of curve_size, at each iteration choose randomly one of m nearest neighbors
    # input -
    # pc: Bx3xN
    # anchor: Bx1
    # m: int -> number of neighbors to choose from
    # curve_size: int
    # output -
    # curve: Bx3xK

    B, num_dims, N = pc.shape
    actual_curve_size = min(curve_size, N)

    neighbors_indices, _ = get_knn(pc, m)

    # exclude self-loop
    neighbors_indices = neighbors_indices[:, :, 1:]
    m = m - 1

    # generate neighbors mask - to choose from at each iteration
    neighbors_ones = torch.ones_like(neighbors_indices)
    neighbors_mask = torch.zeros((B, N, N), device=pc.device).to(neighbors_ones.dtype)
    neighbors_mask = neighbors_mask.scatter_(-1, neighbors_indices, neighbors_ones)

    anchor_for_gather = anchor.unsqueeze(-1).repeat(1, 3, 1)
    curve = torch.gather(pc, 2, anchor_for_gather)

    cur_point = anchor.squeeze(-1)
    for step in range(actual_curve_size - 1):
        # choose random index
        random_indices = torch.randperm(m)
        random_neighbor_index = random_indices[0]

        # Fetch the actual point in the random index
        cur_point = cur_point.view(-1, 1, 1).repeat(1, 1, N)
        neighbors_list = torch.gather(neighbors_mask, index=cur_point, dim=1)
        neighbors = torch.nonzero(neighbors_list, as_tuple=True)[2]
        neighbors = neighbors.view(B, m)
        chosen_neighbor = neighbors[:, random_neighbor_index]  # B,

        chosen_neighbor_for_gather = chosen_neighbor.view(-1, 1, 1).repeat(1, 3, 1)
        neighbor_point = torch.gather(pc, index=chosen_neighbor_for_gather, dim=2)

        # concatenate to the accumulative curve
        curve = torch.cat((curve, neighbor_point), dim=-1)
        cur_point = chosen_neighbor
    return curve
