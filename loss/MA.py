import torch
from torch import nn
import torch.nn.functional as F


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
class MA2(nn.Module):
    def __init__(self,num_pos,feat_norm = 'no'):
        super(MA2, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

        self.layer=nn.Linear(210*768,768)

    def forward(self, cls_feature,g_feature,targets,method='eucilidean'):
        if self.feat_norm == 'yes':
            cls_feature = F.normalize(cls_feature, p=2, dim=-1)
            #color_embed = F.normalize(color_embed, p=2, dim=-1)

        # cls_list=cls_feature.chunk(3,0)
        # N = cls_list[0].size(0)

        N = cls_feature.size(0)  # N:96
        id_num = N // 4 // self.num_pos  # id_num:8

        fc = []
        gc = []
        for i in range(id_num):
            fc.append(cls_feature[targets == targets[i * self.num_pos]].mean(0))
            gc.append(g_feature[targets == targets[i * self.num_pos]].mean(0))
        fc=torch.stack(fc)
        gc = torch.stack(gc)
        # dist=torch.cosine_similarity(fc,gc)


        if method=='eucilidean':
            dist=pdist_torch(fc,gc)
            return torch.mean(torch.diag(dist))
            # ret1=torch.mean(torch.diag(dist))
            # dist2=pdist_torch(fc,fc)
            # b=N//4
            # ret2=[]
            # for i in range(b):
            #     ret2.append(dist2[i,i+b])
            #     ret2.append(dist2[i+2*b, i + 3*b])
            # ret2=torch.mean(torch.stack(ret2))
            # return ret1 #+ret2
        elif method=='cosine':
            dist = torch.cosine_similarity(fc, gc)
            return torch.mean(dist)
        else:
            print("unsupported method!!")
            return None
class MA(nn.Module):
    def __init__(self,num_pos,feat_norm = 'no'):
        super(MA, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

        self.layer=nn.Linear(210*768,768)

    def forward(self, cls_feature,g_feature,targets,method='eucilidean'):
        if self.feat_norm == 'yes':
            cls_feature = F.normalize(cls_feature, p=2, dim=-1)
            #color_embed = F.normalize(color_embed, p=2, dim=-1)

        # cls_list=cls_feature.chunk(3,0)
        # N = cls_list[0].size(0)

        N = cls_feature.size(0)  # N:96
        id_num = N // 3 // self.num_pos  # id_num:8

        fc = []
        gc = []
        for i in range(id_num):
            fc.append(cls_feature[targets == targets[i * self.num_pos]].mean(0))
            gc.append(g_feature[targets == targets[i * self.num_pos]].mean(0))
        fc=torch.stack(fc)
        gc = torch.stack(gc)
        # dist=torch.cosine_similarity(fc,gc)


        if method=='eucilidean':
            dist=pdist_torch(fc,gc)
            return torch.mean(torch.diag(dist))
        elif method=='cosine':
            dist = torch.cosine_similarity(fc, gc)
            return torch.mean(dist)
        else:
            print("unsupported method!!")
            return None

    # def forward(self, cls_feature, g_feature, targets):
    #     if self.feat_norm == 'yes':
    #         cls_feature = F.normalize(cls_feature, p=2, dim=-1)
    #         # color_embed = F.normalize(color_embed, p=2, dim=-1)
    #
    #     # cls_list=cls_feature.chunk(3,0)
    #     # N = cls_list[0].size(0)
    #     dist = torch.cosine_similarity(cls_feature, g_feature)
    #     return torch.mean(dist)


        # N=target_list[0].size(0)
        # is_pos = target_list[0].expand(N, N).eq(target_list[0].expand(N, N).t())
        #
        # dist_g_r=pdist_torch(input_list[1],input_list[0])
        # dist_g_r = dist_g_r[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        # dist_g_r=torch.mean(dist_g_r,dim=1)
        #
        # dist_g_i=pdist_torch(input_list[1],input_list[2])
        # dist_g_i = dist_g_i[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        # dist_g_i= torch.mean(dist_g_i, dim=1)
        #
        # loss = torch.mean(torch.pow(dist_g_r-dist_g_i,2)) / 2
        # return loss




