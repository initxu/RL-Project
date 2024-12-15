import torch
import torch.nn.functional as F
import sys
import pdb
def compute_reward_output(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False, reward_mode=1):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape 
        actions: binary action sequence, shape
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
        reward_mode: 1 for both, 2 for only diversity reward, 3 for only representativeness reward
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)    # seq_len

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(_seq, _seq.t(), beta=1, alpha=-2)
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(-dist_mat.mean())
    
    
    if reward_mode==2:
        reward_rep=0
    elif reward_mode==3:
        reward_div=0
    

    return reward_div,reward_rep


def compute_semantic_reward(seq, capsum_embed, actions, use_gpu=False):
    """
    Compute the semantic reward based on the features of the frame-level captions and the video-level summary

    Args:
        seq: features of  the frame-level captions
        capsum_embed: features of  the video-level summary
        actions: binary action sequence, shape (1, seq_len, 1)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)    # seq_len

    # compute semantic reward
    generated_summary=_seq[pick_idxs]   # [seq_len, embedding_dim]
    capsum_embed = capsum_embed.unsqueeze(0)  # (1, dim)

    # compute similarities
    similarities = F.cosine_similarity(generated_summary, capsum_embed, dim=1)  # (seq_len,)

    reward = torch.mean(similarities)

    return reward
