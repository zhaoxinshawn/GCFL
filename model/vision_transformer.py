import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from itertools import repeat
import collections.abc as container_abcs

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) #[128, 211, 768]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x #64,211,768
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
def pdist_cosine(emb1, emb2):
    '''
    compute the cosine distance matrix between embeddings1 and embeddings2
    using gpu
    '''

    m, n = emb1.shape[0], emb2.shape[0]
    dist_mtx = []
    for i in range(emb1.shape[0]):
        feat_list=[]
        for j in range(emb2.shape[0]):
            x1=emb1[i,:].view(1,-1)
            x2=emb2[j,:].view(1,-1)
            feat_list.append(torch.cosine_similarity(x1,x2))
        feat_list=torch.Tensor(feat_list)
        feat_list=feat_list/torch.sum(feat_list,0)
        dist_mtx.append(feat_list)

    dist_mtx = torch.stack(dist_mtx)
    return dist_mtx

class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches"""
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape  #batch_size , channels , height ,width
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # PatchEmbed_overlap(
        #     (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(12, 12))
        # )
        x = self.proj(x)
        #x:(64,768,21,10)
        x = x.flatten(2).transpose(1, 2)

        return x
class Match_strategy(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sigmoid=nn.Sigmoid()

        self.layer1=nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.layer3=nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
    def forward(self, x,dist_fun=pdist_torch):

        x_patch=x

        avg_x=torch.mean(x_patch,dim=1)
        A_x=torch.mul(avg_x.unsqueeze(1),self.sigmoid(self.norm1(x_patch)))
        # A_x=torch.mul(x_patch,self.sigmoid(self.norm1(A_x)))

        m_x=[]
        for idx in range(A_x.shape[0]):
            m_x.append(dist_fun(A_x[idx,:,:],A_x[idx,:,:]).cuda())
        m_x=torch.stack(m_x,dim=0)
        m_x=torch.max(m_x,dim=-1).values
        m_x =m_x.unsqueeze(-1)
        m_x = torch.add(x_patch, m_x)

        m_x=torch.cat((A_x,m_x),dim=-1)

        m_x=self.layer1(m_x)
        W=self.layer2(m_x)
        Wdyn=torch.mul(x,W)
        x=self.layer4(torch.mul(x,self.layer3(torch.mul(m_x,Wdyn)) ))

        return x.float()#torch.stack([x_cls,x_patch_],dim=1)

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer =nn.LayerNorm):
        super(ViT, self).__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features2(self, x):
        B = x.shape[0]#B:32
        x = self.patch_embed(x) # 64,210,768
        #x:(32,210,768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]
    def forward(self,x):
        cls=self.forward_features2(x)
        return cls

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
class ViT_tri(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer = nn.LayerNorm):
        super(ViT_tri, self).__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.layer=nn.Linear(num_patches*embed_dim,embed_dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features2(self, x,color_comp=False):
        B = x.shape[0]#B:32
        x = self.patch_embed(x) # 64,210,768


        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)


        return x[:, 0]
    def forward(self,x):
        # x = self.forward_features(x)
        # return x
        cls=self.forward_features2(x)
        return cls

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
class ViT_color(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer = nn.LayerNorm):
        super(ViT_color, self).__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.color_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.color_embed, std=.02)
        self.apply(self._init_weights)
        # self.layer=nn.Linear(num_patches*embed_dim,embed_dim)

        self.layer = nn.Sequential(
            nn.Linear((num_patches+1) * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.01)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.01)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features2(self, x,color_comp=False):
        B = x.shape[0]#B:32
        x = self.patch_embed(x) # 64,210,768
        #x:(32,210,768)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        color_embed = self.color_embed.expand(B, -1, -1)
        if color_comp:
            x = x + color_embed
        else:
            x = x - color_embed


        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # g_feat=self.layer(self.color_embed.view(1,-1))
        # if color_comp:
        #     g_feat = x[:, 0] - g_feat
        # else:
        #     g_feat = x[:, 0] + g_feat
        ce = self.layer(self.color_embed.view(1, -1))
        if color_comp:
            g_feat = self.layer2(x[:, 0] - ce)#g_feat
        else:
            g_feat = self.layer2(x[:, 0] + ce)
        return x[:, 0],x[:,1:],g_feat.float()
    def forward(self,x,color_comp=False):
        # x = self.forward_features(x)
        # return x
        cls,patchs,g_feat=self.forward_features2(x,color_comp)
        return cls, patchs,g_feat

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
class ViT_cross(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer = nn.LayerNorm):
        super(ViT_cross, self).__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.color_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.color_embed, std=.02)
        self.apply(self._init_weights)
        # self.layer=nn.Linear(num_patches*embed_dim,embed_dim)

        self.layer = nn.Sequential(
            nn.Linear((num_patches+1) * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.01)
        )
        self.matchStrategy=Match_strategy(dim=embed_dim,drop=drop_rate)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features2(self, x,color_comp=False):
        B = x.shape[0]#B:32
        x = self.patch_embed(x) # 64,210,768
        #x:(32,210,768)

        x = self.matchStrategy(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        color_embed = self.color_embed.expand(B, -1, -1)

        if color_comp:
            x = x + color_embed
        else:
            x = x - color_embed


        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        #
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        # g_feat=self.layer(self.color_embed.view(1,-1))
        # if color_comp:
        #     g_feat = x[:, 0] - g_feat
        # else:
        #     g_feat = x[:, 0] + g_feat
        ce = self.layer(self.color_embed.view(1, -1))
        # if color_comp:
        #     g_feat = self.layer2(x[:, 0] - ce)#g_feat
        # else:
        #     g_feat = self.layer2(x[:, 0] + ce)
        return x,ce
    def forward(self,x,color_comp=False):
        # x = self.forward_features(x)
        # return x
        patchs,ce=self.forward_features2(x,color_comp)
        return patchs,ce

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
class ViT_part(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer = nn.LayerNorm):
        super(ViT_part, self).__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.color_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.num_patches=num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.color_embed, std=.02)
        self.apply(self._init_weights)
        # self.layer=nn.Linear(num_patches*embed_dim,embed_dim)

        self.layer = nn.Sequential(
            nn.Linear((num_patches+1) * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.01)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.01)
        )

        self.part_Layer=nn.ModuleList([])
        for _ in range(num_patches):
            self.part_Layer.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(0.01)
                )
            )
        self.concatlayer=nn.Linear((num_patches+1) * embed_dim, embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def forward_features(self, x):
    #     B = x.shape[0]#B:32
    #     x = self.patch_embed(x) # 64,210,768
    #     #x:(32,210,768)
    #     cls_tokens = self.cls_token.expand(B, -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #
    #     x = x + self.pos_embed
    #
    #     x = self.pos_drop(x)
    #
    #     for blk in self.blocks:
    #         x = blk(x)
    #
    #     x = self.norm(x)
    #
    #     return x[:, 0]
    def forward_features2(self, x,color_comp=False):
        B = x.shape[0]#B:32
        x = self.patch_embed(x) # 64,210,768
        #x:(32,210,768)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        color_embed = self.color_embed.expand(B, -1, -1)
        if color_comp:
            x = x + color_embed
        else:
            x = x - color_embed


        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # g_feat=self.layer(self.color_embed.view(1,-1))
        # if color_comp:
        #     g_feat = x[:, 0] - g_feat
        # else:
        #     g_feat = x[:, 0] + g_feat
        ce = self.layer(self.color_embed.view(1, -1))
        partList=[]
        partList.append(x[:,0])
        for idx in range(self.num_patches):
            tmp=self.part_Layer[idx](x[:,idx+1])
            partList.append(torch.mul(tmp,x[:,0]))
        cls_parts=self.concatlayer(torch.stack(partList,dim=1).view(B,-1))


        if color_comp:
            g_feat = self.layer2(x[:,0] - ce)#g_feat
        else:
            g_feat = self.layer2(x[:,0] + ce)
        return x[:,0].float(),x[:,1:],g_feat.float(),cls_parts.float()
    def forward(self,x,color_comp=False):
        # x = self.forward_features(x)
        # return x
        cls,patchs,g_feat,cls_parts=self.forward_features2(x,color_comp)
        return cls_parts, patchs,g_feat,cls

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))

# class ViT_ce(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer = nn.LayerNorm):
#         super(ViT_color, self).__init__()
#         self.num_classes = num_classes
#
#         self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,embed_dim=embed_dim)
#
#         num_patches = self.patch_embed.num_patches
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#
#         self.color_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)])
#
#         self.norm = norm_layer(embed_dim)
#
#         trunc_normal_(self.cls_token, std=.02)
#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.color_embed, std=.02)
#         self.apply(self._init_weights)
#         self.layer=nn.Linear(num_patches*embed_dim,embed_dim)
#
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     # def forward_features(self, x):
#     #     B = x.shape[0]#B:32
#     #     x = self.patch_embed(x) # 64,210,768
#     #     #x:(32,210,768)
#     #     cls_tokens = self.cls_token.expand(B, -1, -1)
#     #     x = torch.cat((cls_tokens, x), dim=1)
#     #
#     #     x = x + self.pos_embed
#     #
#     #     x = self.pos_drop(x)
#     #
#     #     for blk in self.blocks:
#     #         x = blk(x)
#     #
#     #     x = self.norm(x)
#     #
#     #     return x[:, 0]
#     def forward_features2(self, x,color_comp=False):
#         B = x.shape[0]#B:32
#         x = self.patch_embed(x) # 64,210,768
#         #x:(32,210,768)
#
#         color_embed = self.color_embed.expand(B, -1, -1)
#         if color_comp:
#             x=x+color_embed
#         else:
#             x=x-color_embed
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         x = x + self.pos_embed
#
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         x = self.norm(x)
#
#         g_feat=self.layer(self.color_embed.view(1,-1))
#         if color_comp:
#             g_feat = x[:, 0] - g_feat
#         else:
#             g_feat = x[:, 0] + g_feat
#         return x[:, 0],x[:,1:],g_feat
#     def forward(self,x,color_comp=False):
#         # x = self.forward_features(x)
#         # return x
#         cls,patchs,g_feat=self.forward_features2(x,color_comp)
#         return cls, patchs,g_feat
#
#     def load_param(self, model_path):
#         param_dict = torch.load(model_path, map_location='cpu')
#         if 'model' in param_dict:
#             param_dict = param_dict['model']
#         if 'state_dict' in param_dict:
#             param_dict = param_dict['state_dict']
#         for k, v in param_dict.items():
#             if 'head' in k or 'dist' in k:
#                 continue
#             if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
#                 # For old models that I trained prior to conv based patchification
#                 O, I, H, W = self.patch_embed.proj.weight.shape
#                 v = v.reshape(O, -1, H, W)
#             elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
#                 # To resize pos embedding when using model at different size from pretrained weights
#                 if 'distilled' in model_path:
#                     print('distill need to choose right cls token in the pth')
#                     v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
#                 v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
#             try:
#                 self.state_dict()[k].copy_(v)
#             except:
#                 print('===========================ERROR=========================')
#                 print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):   #标准化
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor