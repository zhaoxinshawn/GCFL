from model.vision_transformer import ViT, ViT_color, ViT_tri, ViT_part, DropPath, Mlp, ViT_cross
import torch
import torch.nn as nn


# L2 norm
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_vision_transformer_BothModality(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_BothModality, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H, cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):
        cls_feature = self.base(x)  # x: 64,3,256,128
        # cls_feature 32,768 ,patch_features 32,210,768
        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature

        else:
            return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_vision_transformer_TM(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_TM, self).__init__()
        self.in_planes = 768

        self.base = ViT_tri(img_size=[cfg.H, cfg.W],
                            stride_size=cfg.STRIDE_SIZE,
                            drop_path_rate=cfg.DROP_PATH,
                            drop_rate=cfg.DROP_OUT,
                            attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):
        cls_feature = self.base(x)  # x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature

        else:
            return self.l2norm(feat)

    # def forward(self, x):
    #     features = self.base(x) #x: 64,3,256,128
    #     feat = self.bottleneck(features)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, features
    #
    #     else:
    #         return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_vision_transformer_TM_CE(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_TM_CE, self).__init__()
        self.in_planes = 768

        self.base = ViT_color(img_size=[cfg.H, cfg.W],
                              stride_size=cfg.STRIDE_SIZE,
                              drop_path_rate=cfg.DROP_PATH,
                              drop_rate=cfg.DROP_OUT,
                              attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

        # self.graycov=nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )
        # self.colorcov = nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )
        # self.conv= nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )

    def forward(self, x):
        # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768

        x_list = x.chunk(3, 0)
        cls_feature0, patch_features0, g_feature0 = self.base(x_list[0], True)
        # x1=self.graycov(x_list[1])
        # x0=self.colorcov(x_list[0])
        # c_modality=self.conv(x0.mul(x1))
        # # if x1.shape[0]!=x2.shape[0]:
        # #     print("\nx1:"+x1.shape+"|x2:"+x2.shape)
        # cls_feature1, patch_features1, g_feature1 = self.base(c_modality)
        cls_feature1, patch_features1, g_feature1 = self.base(x_list[1])
        cls_feature2, patch_features2, g_feature2 = self.base(x_list[2])

        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2])
        patch_features = torch.cat([patch_features0, patch_features1, patch_features2])
        g_feature = torch.cat([g_feature0, g_feature1, g_feature2])

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature, cls_feature, g_feature

        else:
            return self.l2norm(feat)

    # def forward(self, x):
    #     features = self.base(x) #x: 64,3,256,128
    #     feat = self.bottleneck(features)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, features
    #
    #     else:
    #         return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_VI_two_Modality(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_VI_two_Modality, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H, cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):

        x_list = x.chunk(2, 0)
        cls_feature0 = self.base(x_list[0])
        cls_feature1 = self.base(x_list[1])

        cls_feature = torch.cat([cls_feature0, cls_feature1])

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature

        else:
            return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_1gray_vision_transformer(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_1gray_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H, cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):

        x_list = x.chunk(3, 0)
        cls_feature0 = self.base(x_list[0])
        cls_feature1 = self.base(x_list[1])
        cls_feature2 = self.base(x_list[2])

        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2])

        feat = self.bottleneck(cls_feature)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, cls_feature
        else:
            return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_2gray_vision_transformer(nn.Module):

    def __init__(self, num_classes, cfg):
        super(build_2gray_vision_transformer, self).__init__()
        self.in_planes = 768

        self.base = ViT(img_size=[cfg.H, cfg.W],
                        stride_size=cfg.STRIDE_SIZE,
                        drop_path_rate=cfg.DROP_PATH,
                        drop_rate=cfg.DROP_OUT,
                        attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):

        x_list = x.chunk(4, 0)
        cls_feature0 = self.base(x_list[0])
        cls_feature1 = self.base(x_list[1])
        cls_feature2 = self.base(x_list[2])
        cls_feature3 = self.base(x_list[3])

        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2, cls_feature3])

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature

        else:
            return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_2gray_vision_transformer_CE(nn.Module):
    def __init__(self, num_classes, cfg):
        super(build_2gray_vision_transformer_CE, self).__init__()
        self.in_planes = 768

        self.base = ViT_color(img_size=[cfg.H, cfg.W],
                              stride_size=cfg.STRIDE_SIZE,
                              drop_path_rate=cfg.DROP_PATH,
                              drop_rate=cfg.DROP_OUT,
                              attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)
        self.visualnorm = Normalize(2)

    def forward(self, x):
        # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768
        if self.training:
            x_list = x.chunk(4, 0)
            cls_feature0, patch_features0, g_feature0 = self.base(x_list[0])

            cls_feature1, patch_features1, g_feature1 = self.base(x_list[1], True)
            cls_feature2, patch_features2, g_feature2 = self.base(x_list[2], True)
            cls_feature3, patch_features3, g_feature3 = self.base(x_list[3])

            cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2, cls_feature3])
            # patch_features = torch.cat([patch_features0, patch_features1, patch_features2, patch_features3])
            g_feature = torch.cat([g_feature0, g_feature1, g_feature2, g_feature3])
        else:
            cls_feature, patch_features, _ = self.base(x)

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature, g_feature

        else:

            return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_vision_transformer_TM_CE2(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_TM_CE2, self).__init__()
        self.in_planes = 768

        self.base = ViT_color(img_size=[cfg.H, cfg.W],
                              stride_size=cfg.STRIDE_SIZE,
                              drop_path_rate=cfg.DROP_PATH,
                              drop_rate=cfg.DROP_OUT,
                              attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

        # self.graycov=nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )
        # self.colorcov = nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )
        # self.conv= nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )

    def forward(self, x):
        # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768

        x_list = x.chunk(4, 0)
        cls_feature0, patch_features0, g_feature0 = self.base(x_list[0], True)
        # x1=self.graycov(x_list[1])
        # x0=self.colorcov(x_list[0])
        # c_modality=self.conv(x0.mul(x1))
        # # if x1.shape[0]!=x2.shape[0]:
        # #     print("\nx1:"+x1.shape+"|x2:"+x2.shape)
        # cls_feature1, patch_features1, g_feature1 = self.base(c_modality)
        cls_feature1, patch_features1, g_feature1 = self.base(x_list[1])
        cls_feature2, patch_features2, g_feature2 = self.base(x_list[2])
        cls_feature3, patch_features3, g_feature3 = self.base(x_list[3])

        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2, cls_feature3])
        # patch_features = torch.cat([patch_features0, patch_features1, patch_features2, patch_features3])
        g_feature = torch.cat([g_feature0, g_feature1, g_feature2, g_feature3])

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature, cls_feature, g_feature

        else:
            return self.l2norm(feat)

    # def forward(self, x):
    #     features = self.base(x) #x: 64,3,256,128
    #     feat = self.bottleneck(features)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, features
    #
    #     else:
    #         return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


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
        feat_list = []
        for j in range(emb2.shape[0]):
            x1 = emb1[i, :].view(1, -1)
            x2 = emb2[j, :].view(1, -1)
            feat_list.append(torch.cosine_similarity(x1, x2))
        feat_list = torch.Tensor(feat_list)
        feat_list = feat_list / torch.sum(feat_list, 0)
        dist_mtx.append(feat_list)

    dist_mtx = torch.stack(dist_mtx)
    return dist_mtx


class build_vision_transformer_TM_CC(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_TM_CC, self).__init__()
        self.in_planes = 768

        self.base = ViT_color(img_size=[cfg.H, cfg.W],
                              stride_size=cfg.STRIDE_SIZE,
                              drop_path_rate=cfg.DROP_PATH,
                              drop_rate=cfg.DROP_OUT,
                              attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):
        # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768

        x_list = x.chunk(3, 0)
        cls_feature0, patch_features0, g_feature0 = self.base(x_list[0])
        cls_feature1, patch_features1, g_feature1 = self.base(x_list[1], True)
        cls_feature2, patch_features2, g_feature2 = self.base(x_list[2], True)

        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2])
        patch_features = torch.cat([patch_features0, patch_features1, patch_features2])
        color_embed = torch.cat([g_feature0, g_feature1, g_feature2])

        feat = self.bottleneck(cls_feature)
        feat_cc = self.bottleneck(color_embed)

        if self.training:
            cls_score = self.classifier(feat)
            cc_score = self.classifier(feat_cc)

            return cls_score, cls_feature, cc_score, color_embed

        else:
            return self.l2norm(feat)

    # def forward(self, x):
    #     features = self.base(x) #x: 64,3,256,128
    #     feat = self.bottleneck(features)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, features
    #
    #     else:
    #         return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class cross_Attention(nn.Module):
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

    def forward(self, x, y):
        B, N, C = x.shape
        x_qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_k, x_v = x_qkv[0], x_qkv[1], x_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # [43, 211, 3, 12, 64]
        # if False==self.training:
        #     print(">>>>>>"+self.training)

        y_qkv = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_q, y_k, y_v = y_qkv[0], y_qkv[1], y_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        x_attn = (y_q @ x_k.transpose(-2, -1)) * self.scale
        x_attn = x_attn.softmax(dim=-1)
        x_attn = self.attn_drop(x_attn)

        x = (x_attn @ x_v).transpose(1, 2).reshape(B, N, C)  # [128, 211, 768]
        x = self.proj(x)
        x = self.proj_drop(x)

        # y_attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        # y_attn = y_attn.softmax(dim=-1)
        # y_attn = self.attn_drop(y_attn)
        #
        # y = (y_attn @ y_v).transpose(1, 2).reshape(B, N, C)  # [128, 211, 768]
        # y = self.proj(y)
        # y = self.proj_drop(y)

        return x, y


class cross_Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = cross_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        xx, yy = self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = self.norm2(x + xx)
        y = self.norm2(y + yy)
        x = self.norm3(x + self.drop_path(self.mlp(x)))
        y = self.norm3(y + self.drop_path(self.mlp(y)))
        return x, y  # 64,211,768


class build_vision_transformer_crossBlocks(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_crossBlocks, self).__init__()
        self.in_planes = 768

        self.base = ViT_cross(img_size=[cfg.H, cfg.W],
                              stride_size=cfg.STRIDE_SIZE,
                              drop_path_rate=cfg.DROP_PATH,
                              drop_rate=cfg.DROP_OUT,
                              attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)
        self.embed_dim = self.in_planes
        self.depth = 12
        # dpr = [x.item() for x in torch.linspace(0, cfg.DROP_PATH, self.depth)]
        # self.cross_blocks = nn.ModuleList([
        #     cross_Block(
        #         dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #         drop=cfg.DROP_OUT, attn_drop=cfg.ATT_DROP_RATE, drop_path=dpr[i], norm_layer=nn.LayerNorm)
        #     for i in range(self.depth)])
        # self.cross_blocks =cross_Block(
        #             dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #             drop=cfg.DROP_OUT, attn_drop=cfg.ATT_DROP_RATE, drop_path=dpr[0], norm_layer=nn.LayerNorm)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

        self.layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(0.01)
        )

    def forward(self, x):
        # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768

        x_list = x.chunk(3, 0)
        patchs0, ce0 = self.base(x_list[0], True)

        # for idx,blk in enumerate(self.cross_blocks):
        #     if idx>=6:
        #         break
        #     x_list[1], x_list[2] = blk(x_list[1], x_list[2])
        patchs1, ce1 = self.base(x_list[1])
        patchs2, ce2 = self.base(x_list[2])

        # if True==self.training:
        #     #patchs0, patchs1 = self.cross_blocks(patchs1, patchs2)
        #     patchs1,patchs0=self.cross_blocks(patchs1,patchs0)
        # patchs0, patchs0 = self.cross_blocks(patchs0, patchs0)
        # patchs1, patchs1 = self.cross_blocks(patchs1, patchs1)
        # patchs2, patchs2 = self.cross_blocks(patchs2, patchs2)

        cls_feature0 = patchs0[:, 0]
        cls_feature1 = patchs1[:, 0]
        cls_feature2 = patchs2[:, 0]
        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2])

        g_feature0 = self.layer(patchs0[:, 0] - ce0)
        g_feature1 = self.layer(patchs0[:, 1] + ce1)
        g_feature2 = self.layer(patchs0[:, 2] + ce2)

        g_feature = torch.cat([g_feature0.float(), g_feature1.float(), g_feature2.float()])

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature, g_feature

        else:
            return self.l2norm(feat)

    # def forward(self, x):
    #     features = self.base(x) #x: 64,3,256,128
    #     feat = self.bottleneck(features)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, features
    #
    #     else:
    #         return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_vision_transformer_matchStrategy(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_TM_CE, self).__init__()
        self.in_planes = 768

        self.base = ViT_color(img_size=[cfg.H, cfg.W],
                              stride_size=cfg.STRIDE_SIZE,
                              drop_path_rate=cfg.DROP_PATH,
                              drop_rate=cfg.DROP_OUT,
                              attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x):
        # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768

        x_list = x.chunk(3, 0)
        cls_feature0, patch_features0, g_feature0 = self.base(x_list[0], True)
        # x1=self.graycov(x_list[1])
        # x0=self.colorcov(x_list[0])
        # c_modality=self.conv(x0.mul(x1))
        # # if x1.shape[0]!=x2.shape[0]:
        # #     print("\nx1:"+x1.shape+"|x2:"+x2.shape)
        # cls_feature1, patch_features1, g_feature1 = self.base(c_modality)
        cls_feature1, patch_features1, g_feature1 = self.base(x_list[1])
        cls_feature2, patch_features2, g_feature2 = self.base(x_list[2])

        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2])
        patch_features = torch.cat([patch_features0, patch_features1, patch_features2])
        g_feature = torch.cat([g_feature0, g_feature1, g_feature2])

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature, cls_feature, g_feature

        else:
            return self.l2norm(feat)

    # def forward(self, x):
    #     features = self.base(x) #x: 64,3,256,128
    #     feat = self.bottleneck(features)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, features
    #
    #     else:
    #         return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


# class build_vision_transformer_crossBlocks(nn.Module):
#     #colorEmbedding
#     def __init__(self, num_classes, cfg):
#         super(build_vision_transformer_crossBlocks, self).__init__()
#         self.in_planes = 768
#         # self, img_size = 224, patch_size = 16, stride_size = 16, in_chans = 3, num_classes = 1000, embed_dim = 768, depth = 12,
#         # num_heads = 12, mlp_ratio = 4., qkv_bias = True, qk_scale = None, drop_rate = 0., attn_drop_rate = 0.,
#         # drop_path_rate = 0., norm_layer = nn.LayerNorm
#         self.base = ViT_color(img_size=[cfg.H,cfg.W],
#                         stride_size=cfg.STRIDE_SIZE,
#                         drop_path_rate=cfg.DROP_PATH,
#                         drop_rate=cfg.DROP_OUT,
#                         attn_drop_rate=cfg.ATT_DROP_RATE)
#         self.deep=12
#         dpr = [x.item() for x in torch.linspace(0, cfg.DROP_PATH, self.deep)]
#         self.cross_blocks = nn.ModuleList([
#             cross_Block(
#                 dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                 drop=cfg.DROP_OUT, attn_drop=cfg.ATT_DROP_RATE, drop_path=dpr[i], norm_layer=nn.LayerNorm)
#             for i in range(self.depth)])
#         # self.cross_block=cross_Block(
#         #     drop=cfg.DROP_OUT, attn_drop=cfg.ATT_DROP_RATE, drop_path=dpr[0], norm_layer=nn.LayerNorm
#         # )
#         self.base.load_param(cfg.PRETRAIN_PATH)
#
#         print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))
#
#         self.num_classes = num_classes
#
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#         self.classifier.apply(weights_init_classifier)
#
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)
#
#         self.l2norm = Normalize(2)
#
#
#     def forward(self, x):
#         # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
#         # #cls_feature 32,768 ,patch_features 32,210,768
#
#         x_list=x.chunk(3,0)
#         cls_feature0, patch_features0,g_feature0 = self.base(x_list[0])
#         x_list[1],x_list[2]=self.cross_block(x_list[1],x_list[2])
#         cls_feature1, patch_features1,g_feature1 = self.base(x_list[1],True)
#         cls_feature2, patch_features2, g_feature2 = self.base(x_list[1], True)
#
#         cls_feature=torch.cat([cls_feature0,cls_feature1,cls_feature2])
#         patch_features=torch.cat([patch_features0,patch_features1,patch_features2])
#         color_embed = torch.cat([g_feature0, g_feature1, g_feature2])
#
#         feat = self.bottleneck(cls_feature)
#         feat_cc = self.bottleneck(color_embed)
#
#         if self.training:
#             cls_score = self.classifier(feat)
#             cc_score = self.classifier(feat_cc)
#
#             return cls_score, cls_feature,cc_score,color_embed
#
#         else:
#             return self.l2norm(feat)
#     # def forward(self, x):
#     #     features = self.base(x) #x: 64,3,256,128
#     #     feat = self.bottleneck(features)
#     #
#     #     if self.training:
#     #         cls_score = self.classifier(feat)
#     #
#     #         return cls_score, features
#     #
#     #     else:
#     #         return self.l2norm(feat)
#
#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path)
#         for i in param_dict:
#             self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#         print('Loading pretrained model from {}'.format(trained_path))
#
#     def load_param_finetune(self, model_path):
#         param_dict = torch.load(model_path)
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])
#         print('Loading pretrained model for finetuning from {}'.format(model_path))
class build_vision_transformer_part(nn.Module):
    # colorEmbedding
    def __init__(self, num_classes, cfg):
        super(build_vision_transformer_part, self).__init__()
        self.in_planes = 768

        self.base = ViT_part(img_size=[cfg.H, cfg.W],
                             stride_size=cfg.STRIDE_SIZE,
                             drop_path_rate=cfg.DROP_PATH,
                             drop_rate=cfg.DROP_OUT,
                             attn_drop_rate=cfg.ATT_DROP_RATE)

        self.base.load_param(cfg.PRETRAIN_PATH)

        print('Loading pretrained ImageNet model......from {}'.format(cfg.PRETRAIN_PATH))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

        # self.graycov=nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )
        # self.colorcov = nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )
        # self.conv= nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1), bias=False),
        #     nn.ReLU()
        # )

    def forward(self, x):
        # cls_feature,patch_features = self.base(x) #x: 64,3,256,128
        # #cls_feature 32,768 ,patch_features 32,210,768

        x_list = x.chunk(3, 0)
        cls_feature0, patch_features0, g_feature0, cls_patch0 = self.base(x_list[0], True)
        # x1=self.graycov(x_list[1])
        # x0=self.colorcov(x_list[0])
        # c_modality=self.conv(x0.mul(x1))
        # # if x1.shape[0]!=x2.shape[0]:
        # #     print("\nx1:"+x1.shape+"|x2:"+x2.shape)
        # cls_feature1, patch_features1, g_feature1 = self.base(c_modality)
        cls_feature1, patch_features1, g_feature1, cls_patch1 = self.base(x_list[1])
        cls_feature2, patch_features2, g_feature2, cls_patch2 = self.base(x_list[2])

        cls_feature = torch.cat([cls_feature0, cls_feature1, cls_feature2])
        patch_features = torch.cat([patch_features0, patch_features1, patch_features2])
        color_embed = torch.cat([g_feature0, g_feature1, g_feature2])
        cls = torch.cat([cls_patch0, cls_patch1, cls_patch2])

        feat = self.bottleneck(cls_feature)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, cls_feature, patch_features, color_embed, cls

        else:
            return self.l2norm(feat)

    # def forward(self, x):
    #     features = self.base(x) #x: 64,3,256,128
    #     feat = self.bottleneck(features)
    #
    #     if self.training:
    #         cls_score = self.classifier(feat)
    #
    #         return cls_score, features
    #
    #     else:
    #         return self.l2norm(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
