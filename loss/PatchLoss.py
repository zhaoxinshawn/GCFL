import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
def euclidean_dist(x, y, eps=1e-12):
	"""
	Args:
	  x: pytorch Tensor, with shape [m, d]
	  y: pytorch Tensor, with shape [n, d]
	Returns:
	  dist: pytorch Tensor, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(x, y.t(), beta=1, alpha=-2) #dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=eps).sqrt()

	return dist
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def cal_dist(x1,x2,fun=euclidean_dist):
    n=x1.size(0)
    ret=[]
    for i in range(n):
        ret.append(fun(x1[i,:,:],x2[i,:,:]))
    return torch.stack(ret,dim=0)
# def matching_loss()
class PatchLOSS(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(PatchLOSS, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm
    # def comp_loss(self,cls_feature,patch_features,pdist=None):
    #     pB, pN, _ = patch_features.shape
    #
    #     # patch_center = torch.sum(patch_features, dim=1)
    #     # patch_center = torch.div(patch_center, pN)  # 64,768
    #     patch_center = torch.mean(patch_features, dim=1)
    #     dist_cls = pdist(patch_center, cls_feature)
    #
    #     patch_center = torch.unsqueeze(patch_center, dim=1)
    #     dist_pths = []
    #     for i in range(pB):
    #         x1 = patch_center[i, :, :]
    #         x2 = patch_features[i, :, :]
    #         d=pdist(x1, x2)
    #         # mean = torch.mean(d)
    #         # d=d[ d <mean].mean(0)
    #         sort_d,_=torch.sort(d,dim=-1)
    #         d=d[ d < sort_d[9] ].mean(0)
    #         dist_pths.append(d + dist_cls[i])
    #     loss = torch.stack(dist_pths,dim=0).mean(dim=0)
    #     return loss

    def forward(self,patch_features, metric='eucilidean',threshold=1e-6):
        # 96,768   96,210,768 color-gray-thermal

        if self.feat_norm == 'yes':

            patch_features = F.normalize(patch_features, p=2, dim=-1)

        if metric=='eucilidean':
            pdist=euclidean_dist#pdist=nn.PairwiseDistance(p=2)
        elif metric=='cosine':
            pdist=nn.CosineSimilarity(dim=1, eps=1e-6)


        pths_list=patch_features.chunk(3, 0)

        C2T=cal_dist(pths_list[0],pths_list[2])
        G2T=cal_dist(pths_list[1],pths_list[2])

        diff=torch.sub(input=C2T,alpha=1,other=G2T)
        # diff2= C2T - G2T

        diff=torch.abs(diff).view(diff.size(0),-1).mean(1)
        return diff.mean(0)

if __name__ == '__main__':
    loss=PatchLOSS()
    c=torch.randn((96,768))
    xx=torch.randn((96,210,768))
    loss(c,xx)
