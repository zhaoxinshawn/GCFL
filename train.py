import random

from optuna.trial import TrialState
from tqdm import tqdm

from dataloader import SYSUData, RegDBData, TestData, GenIdx, IdentitySampler, SYSUTriData, RegDBTriData, \
    RegDBTri2Data, Tri2Sampler, SYSUTri2Data, TwoMSampler, RegDB_Igary_Data, RegDB_Rgary_Data, Tri1Sampler
from datamanager import process_gallery_sysu, process_query_sysu, process_test_regdb
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch
from torch.cuda import amp
import torch.nn as nn
import os.path as osp
import os

from loss.GCL import GCL, GCL2
from loss.GMSEL import GMSEL, GMSEL2
from loss.MA import MA, MA2
from loss.PatchLoss import PatchLOSS
from model.make_model import build_vision_transformer_TM, build_vision_transformer_TM_CE, \
    build_vision_transformer_TM_CC, build_vision_transformer_part, build_vision_transformer_crossBlocks, \
    build_vision_transformer_TM_CE2, build_VI_two_Modality, build_1gray_vision_transformer, \
    build_2gray_vision_transformer, build_2gray_vision_transformer_CE
import time
import optimizer
from scheduler import create_scheduler
from loss.Triplet import TripletLoss, IntraLoss
from loss.MSEL import MSEL
from loss.DCL import DCL
from utils import AverageMeter, set_seed, Logging, rand_bbox
from transforms import transform_rgb, transform_rgb2gray, transform_thermal, transform_test, transform_regdb, \
    transform_regdbtogray, transform_regdbtogray_grayMix
from optimizer import make_optimizer
from config.config import cfg
from eval_metrics import eval_sysu, eval_regdb
import argparse
from prettytable import PrettyTable
import optuna

parser = argparse.ArgumentParser(description="PMT Training")
parser.add_argument('--config_file', default='config/SYSU.yml',
                    # default='config/SYSU.yml',#default='config/RegDB.yml',
                    help='path to config file', type=str)
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', '-r', default='',
                    help='resume from checkpoint', type=str)
parser.add_argument('--model_path', default='save_model/',
                    help='model save path', type=str)
parser.add_argument('--num_workers', default=0,
                    help='number of data loading workers', type=int)
parser.add_argument('--start_test', default=0,
                    help='start to test in training', type=int)
parser.add_argument('--test_batch', default=128,
                    help='batch size for test', type=int)
parser.add_argument('--test_epoch', default=2,
                    help='test model every 2 epochs', type=int)
parser.add_argument('--save_epoch', default=2,
                    help='save model every 2 epochs', type=int)
parser.add_argument('--gpu', default='0',
                    help='gpu device ids for CUDA_VISIBLE_DEVICES', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if args.config_file != '':
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

set_seed(cfg.SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
logger = Logging(cfg.logger_dir)

# statistics
dataset_statistics = PrettyTable([cfg.DATASET, 'images', 'identities', 'cameras'])
result_statistics = PrettyTable(['epoch', 'mAP', 'mInp', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20'])

if cfg.DATASET == 'sysu':
    data_path = cfg.DATA_PATH_SYSU

    trainset = SYSUTri2Data(data_path,
                            transform_color=transform_rgb,
                            transform_gcolor=transform_rgb2gray,
                            transform_gthermal=transform_rgb2gray,
                            transform_thermal=transform_thermal, )

    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    dataset_statistics.add_row(
        ['train_RGB', len(trainset.train_color_image), len(np.unique(trainset.train_color_label)), 'None']
    )
    dataset_statistics.add_row(
        ['train_IR', len(trainset.train_thermal_image), len(np.unique(trainset.train_thermal_label)), 'None']
    )
elif cfg.DATASET == 'regdb':
    data_path = cfg.DATA_PATH_RegDB

    if cfg.METHOD == 'VI_two_Modality':
        trainset = RegDBData(data_path, args.trial,
                             transform_color=transform_regdb,
                             transform_thermal=transform_thermal)

    elif cfg.METHOD == 'RGB_gary_Modality':
        trainset = RegDB_Rgary_Data(data_path, args.trial,
                                    transform_color=transform_regdb,
                                    transform_gray=transform_rgb2gray,
                                    transform_thermal=transform_thermal)

    elif cfg.METHOD == 'IR_gary_Modality':
        trainset = RegDB_Igary_Data(data_path, args.trial,
                                    transform_color=transform_regdb,
                                    transform_gray=transform_rgb2gray,
                                    transform_thermal=transform_thermal)

    elif cfg.METHOD == 'Tri_2gary_Modality' or cfg.METHOD == 'Tri_2gary_Modality_CE':
        trainset = RegDBTri2Data(data_path, args.trial,
                                 transform_color=transform_regdb,
                                 transform_gcolor=transform_rgb2gray,
                                 transform_gthermal=transform_rgb2gray,
                                 transform_thermal=transform_thermal)

    else:
        trainset = RegDBTri2Data(data_path, args.trial,
                                 transform_color=transform_regdb,
                                 # transform_color=transform_regdb, # transform_color=transform_rgb, # only transform_regdb is best
                                 transform_gcolor=transform_rgb2gray,  # transform_gcolor=transform_rgb2gray,
                                 transform_gthermal=transform_rgb2gray,
                                 transform_thermal=transform_thermal)  # transform_thermal=transform_thermal, )

    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    print('Current trial :', args.trial)
    dataset_statistics.add_row(
        ['train_RGB', len(trainset.train_color_image), len(np.unique(trainset.train_color_label)), 'None']
    )
    dataset_statistics.add_row(
        ['train_IR', len(trainset.train_thermal_image), len(np.unique(trainset.train_thermal_label)), 'None']
    )

num_classes = len(np.unique(trainset.train_color_label))
# model = build_vision_transformer(num_classes=num_classes, cfg=cfg)
if cfg.METHOD == 'VI_two_Modality':
    model = build_VI_two_Modality(num_classes=num_classes, cfg=cfg)
elif cfg.METHOD == 'RGB_gary_Modality' or cfg.METHOD == 'IR_gary_Modality':
    model = build_1gray_vision_transformer(num_classes=num_classes, cfg=cfg)
elif cfg.METHOD == 'Tri_2gary_Modality':
    model = build_2gray_vision_transformer(num_classes=num_classes, cfg=cfg)
elif cfg.METHOD == 'Tri_2gary_Modality_CE':
    model = build_2gray_vision_transformer_CE(num_classes=num_classes, cfg=cfg)

elif cfg.METHOD == 'TripletModality_colorEmbedding2':
    model = build_vision_transformer_TM_CE2(num_classes=num_classes, cfg=cfg)
else:
    model = build_vision_transformer_TM_CE2(num_classes=num_classes, cfg=cfg)
# model.to(device)

# load checkpoint
if len(args.resume) > 0:
    model_path = args.model_path + args.resume + '.pth'
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        model.load_param(model_path)
        print('==> loaded checkpoint {}'.format(args.resume))
    else:
        print('==> no checkpoint found at {}'.format(model_path))
# model = nn.DataParallel(model).to(device)
model.to(device)

# Loss
criterion_ID = nn.CrossEntropyLoss()
criterion_Tri = TripletLoss(margin=cfg.MARGIN, feat_norm='no')
# criterion_Intra = IntraLoss(feat_norm='no')
# criterion_GCL =DCL(num_pos=cfg.NUM_POS, feat_norm='no')
# criterion_GMSEL = MSEL(num_pos=cfg.NUM_POS, feat_norm='no')
if cfg.METHOD == 'RGB_gary_Modality' or cfg.METHOD == 'IR_gary_Modality':
    criterion_GCL = GCL(num_pos=cfg.NUM_POS, feat_norm='no')
    criterion_GMSEL = GMSEL(num_pos=cfg.NUM_POS, feat_norm='no')
    criterion_MA = MA(num_pos=cfg.NUM_POS, feat_norm='no')
elif cfg.METHOD == 'Tri_2gary_Modality' or cfg.METHOD == 'Tri_2gary_Modality_CE':
    criterion_GCL = GCL2(num_pos=cfg.NUM_POS, feat_norm='no')
    criterion_GMSEL = GMSEL2(num_pos=cfg.NUM_POS, feat_norm='no')
    criterion_MA = MA2(num_pos=cfg.NUM_POS, feat_norm='no')
else:
    criterion_GCL = None
    criterion_GMSEL = None

# criterion_PL = PatchLOSS(num_pos=cfg.NUM_POS, feat_norm='no')

optimizer = make_optimizer(cfg, model)
scheduler = create_scheduler(cfg, optimizer)

scaler = amp.GradScaler()

# for test
if cfg.DATASET == 'sysu':
    assert cfg.mode == 'all' or cfg.mode == 'indoor', 'cfg.mode: ' + cfg.mode + 'not in [all,indoor]'

    if cfg.mode == 'all':
        query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')  # mode='all',#mode='indoor',
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0,
                                                              gall_mode='single')  # mode='all',
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    else:
        query_img, query_label, query_cam = process_query_sysu(data_path, mode='indoor')  # mode='all',#mode='indoor',
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='indoor', trial=0,
                                                              gall_mode='single')  # mode='all',
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    # print dataset_statistics
    dataset_statistics.add_row(['query', len(query_img), len(np.unique(query_label)), len(np.unique(query_cam))])
    dataset_statistics.add_row(['gallery', len(gall_img), len(np.unique(gall_label)), len(np.unique(gall_cam))])
elif cfg.DATASET == 'regdb':
    assert cfg.mode == 'v2t' or cfg.mode == 't2v', 'cfg.mode: ' + cfg.mode + 'not in [v2t,t2v]'

    if cfg.mode == 'v2t':
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')  # modal='visible')
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')  # , modal='thermal')
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
    else:
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')  # modal='visible')
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))

        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')  # , modal='thermal')
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))

    # print dataset_statistics
    dataset_statistics.add_row(['query', len(query_img), len(np.unique(query_label)), 'None'])
    dataset_statistics.add_row(['gallery', len(gall_img), len(np.unique(gall_label)), 'None'])

logger(str(dataset_statistics))

# Test loader
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)

loss_meter = AverageMeter()
loss_ce_meter = AverageMeter()
loss_tri_meter = AverageMeter()
acc_rgb_meter = AverageMeter()
acc_ir_meter = AverageMeter()


def train(epoch):
    start_time = time.time()

    loss_meter.reset()
    loss_ce_meter.reset()
    loss_tri_meter.reset()
    acc_rgb_meter.reset()
    acc_ir_meter.reset()

    scheduler.step(epoch)
    model.train()
    if cfg.METHOD == 'VI_two_Modality':
        for idx, (input_color, input_thermal, label_color, label_thermal) in enumerate(
                tqdm(trainloader)):
            optimizer.zero_grad()
            input_color = input_color.to(device)
            input_thermal = input_thermal.to(device)
            label_color = label_color.to(device)
            label_thermal = label_thermal.to(device)

            labels = torch.cat((label_color, label_thermal), 0)

            with amp.autocast(enabled=True):
                cls_score, cls_feature = model(torch.cat([input_color, input_thermal]))
                score_color, score_thermal = cls_score.chunk(2, 0)
                feat_color, feat_thermal = cls_feature.chunk(2, 0)

                loss_id = criterion_ID(score_color, label_color.long()) + \
                          criterion_ID(score_thermal, label_thermal.long())

                if epoch <= cfg.PL_EPOCH:
                    loss_tri = criterion_Tri(feat_color, feat_color, label_color) + \
                               criterion_Tri(feat_thermal, feat_thermal, label_thermal)

                    loss = loss_id + loss_tri

                else:
                    loss_tri = criterion_Tri(cls_feature, cls_feature, labels)

                    loss = loss_id + loss_tri

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_rgb = (score_color.max(1)[1] == label_color).float().mean()
            acc_ir = (score_thermal.max(1)[1] == label_thermal).float().mean()

            loss_tri_meter.update(loss_tri.item())
            loss_ce_meter.update(loss_id.item())
            loss_meter.update(loss.item())

            acc_rgb_meter.update(acc_rgb, 1)
            acc_ir_meter.update(acc_ir, 1)

            torch.cuda.synchronize()

            if (idx + 1) % 64 == 0:
                print('Epoch[{}] Iteration[{}/{}]'
                      ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                      'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                      'Base Lr: {:.2e} '.format(epoch, (idx + 1),
                                                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                                                loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                                                optimizer.state_dict()['param_groups'][0]['lr']))
    elif cfg.METHOD == 'RGB_gary_Modality' or cfg.METHOD == 'IR_gary_Modality':
        for idx, (input_color, input_gray, input_thermal,
                  label_color, label_gray, label_thermal) in enumerate(tqdm(trainloader)):
            # img_color, img_gray,img_thermal, target_color, target_gray, target_color
            optimizer.zero_grad()
            input_color = input_color.to(device)
            input_gray = input_gray.to(device)
            input_thermal = input_thermal.to(device)

            label_color = label_color.to(device)
            label_gray = label_gray.to(device)
            label_thermal = label_thermal.to(device)

            labels = torch.cat((label_color, label_gray, label_thermal), 0)

            with amp.autocast(enabled=True):
                cls_score, cls_feature = model(
                    torch.cat([input_color, input_gray, input_thermal])
                )

                score_color, score_gray, score_thermal = cls_score.chunk(3, 0)
                feat_color, feat_gray, feat_thermal = cls_feature.chunk(3, 0)

                loss_id = criterion_ID(score_color, label_color.long()) + \
                          criterion_ID(score_gray, label_gray.long()) + \
                          criterion_ID(score_thermal, label_thermal.long())

                if epoch <= cfg.PL_EPOCH:
                    loss_tri = criterion_Tri(feat_color, feat_color, label_color) + \
                               criterion_Tri(feat_gray, feat_gray, label_gray) + \
                               criterion_Tri(feat_thermal, feat_thermal, label_thermal)

                    loss = loss_id + loss_tri

                else:
                    loss_tri = criterion_Tri(cls_feature, cls_feature, labels)

                    loss_gmsel = criterion_GMSEL(cls_feature, labels)
                    loss_gdcl = criterion_GCL(cls_feature, labels)

                    loss = loss_id + loss_tri + cfg.GMSEL * loss_gmsel + cfg.GDCL * loss_gdcl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_rgb = (score_color.max(1)[1] == label_color).float().mean()
            acc_ir = (score_thermal.max(1)[1] == label_thermal).float().mean()

            loss_tri_meter.update(loss_tri.item())
            loss_ce_meter.update(loss_id.item())
            loss_meter.update(loss.item())

            acc_rgb_meter.update(acc_rgb, 1)
            acc_ir_meter.update(acc_ir, 1)

            torch.cuda.synchronize()

            if (idx + 1) % 64 == 0:
                print('Epoch[{}] Iteration[{}/{}]'
                      ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                      'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                      'Base Lr: {:.2e} '.format(epoch, (idx + 1),
                                                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                                                loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                                                optimizer.state_dict()['param_groups'][0]['lr']))
    elif cfg.METHOD == 'Tri_2gary_Modality':
        for idx, (input_color, input_thermal, input_gcolor, input_gthermal,
                  label_color, label_thermal, label_gcolor, label_gthermal) in enumerate(tqdm(trainloader)):
            # img_color, img_gray,img_thermal, target_color, target_gray, target_color
            optimizer.zero_grad()
            input_color = input_color.to(device)
            input_gcolor = input_gcolor.to(device)
            input_gthermal = input_gthermal.to(device)
            input_thermal = input_thermal.to(device)

            label_color = label_color.to(device)
            label_gcolor = label_gcolor.to(device)
            label_gthermal = label_gthermal.to(device)
            label_thermal = label_thermal.to(device)

            labels = torch.cat((label_color, label_gcolor, label_gthermal, label_thermal), 0)

            with amp.autocast(enabled=True):
                # color_diff=input_color-input_gray

                cls_score, cls_feature = model(
                    torch.cat([input_color, input_gcolor, input_gthermal, input_thermal]))

                score_color, score_gcolor, score_gthermal, score_thermal = cls_score.chunk(4, 0)
                feat_color, feat_gcolor, feat_gthermal, feat_thermal = cls_feature.chunk(4, 0)

                loss_id = criterion_ID(score_color, label_color.long()) + \
                          criterion_ID(score_gcolor, label_gcolor.long()) + \
                          criterion_ID(score_gthermal, label_gthermal.long()) + \
                          criterion_ID(score_thermal, label_thermal.long())

                if epoch <= cfg.PL_EPOCH:
                    loss_tri = criterion_Tri(feat_color, feat_color, label_color) + \
                               criterion_Tri(feat_gcolor, feat_gcolor, label_gcolor) + \
                               criterion_Tri(feat_gthermal, feat_gthermal, label_gthermal) + \
                               criterion_Tri(feat_thermal, feat_thermal, label_thermal)

                    loss = loss_id + loss_tri

                else:
                    loss_tri = criterion_Tri(cls_feature, cls_feature, labels)

                    loss_gmsel = criterion_GMSEL(cls_feature, labels)
                    loss_gdcl = criterion_GCL(cls_feature, labels)

                    loss = loss_id + loss_tri #+ cfg.GMSEL * loss_gmsel + cfg.GDCL * loss_gdcl
            #
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_rgb = (score_color.max(1)[1] == label_color).float().mean()
            acc_ir = (score_thermal.max(1)[1] == label_thermal).float().mean()

            loss_tri_meter.update(loss_tri.item())
            loss_ce_meter.update(loss_id.item())
            loss_meter.update(loss.item())

            acc_rgb_meter.update(acc_rgb, 1)
            acc_ir_meter.update(acc_ir, 1)

            torch.cuda.synchronize()

            if (idx + 1) % 64 == 0:
                print('Epoch[{}] Iteration[{}/{}]'
                      ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                      'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                      'Base Lr: {:.2e} '.format(epoch, (idx + 1),
                                                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                                                loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                                                optimizer.state_dict()['param_groups'][0]['lr']))
    elif cfg.METHOD == 'Tri_2gary_Modality_CE':
        for idx, (input_color, input_thermal, input_gcolor, input_gthermal,
                  label_color, label_thermal, label_gcolor, label_gthermal) in enumerate(tqdm(trainloader)):
            # img_color, img_gray,img_thermal, target_color, target_gray, target_color
            optimizer.zero_grad()
            input_color = input_color.to(device)
            input_gcolor = input_gcolor.to(device)
            input_gthermal = input_gthermal.to(device)
            input_thermal = input_thermal.to(device)

            label_color = label_color.to(device)
            label_gcolor = label_gcolor.to(device)
            label_gthermal = label_gthermal.to(device)
            label_thermal = label_thermal.to(device)

            labels = torch.cat((label_color, label_gcolor, label_gthermal, label_thermal), 0)

            with amp.autocast(enabled=True):
                # color_diff=input_color-input_gray

                cls_score, cls_feature, g_feature = model(
                    torch.cat([input_color, input_gcolor, input_gthermal, input_thermal]))

                score_color, score_gcolor, score_gthermal, score_thermal = cls_score.chunk(4, 0)
                feat_color, feat_gcolor, feat_gthermal, feat_thermal = cls_feature.chunk(4, 0)

                loss_id = criterion_ID(score_color, label_color.long()) + \
                          criterion_ID(score_gcolor, label_gcolor.long()) + \
                          criterion_ID(score_gthermal, label_gthermal.long()) + \
                          criterion_ID(score_thermal, label_thermal.long())

                # loss_ma = criterion_MA(cls_feature, g_feature, labels, method='eucilidean')
                loss_ma = criterion_MA(cls_feature, g_feature, labels, method='cosine')

                if epoch <= cfg.PL_EPOCH:
                    loss_tri = criterion_Tri(feat_color, feat_color, label_color) + \
                               criterion_Tri(feat_gcolor, feat_gcolor, label_gcolor) + \
                               criterion_Tri(feat_gthermal, feat_gthermal, label_gthermal) + \
                               criterion_Tri(feat_thermal, feat_thermal, label_thermal)

                    loss = loss_id + loss_tri + cfg.MA * loss_ma

                else:
                    loss_tri = criterion_Tri(cls_feature, cls_feature, labels)

                    loss_gmsel = criterion_GMSEL(cls_feature, labels,method='eucilidean')
                    loss_gdcl = criterion_GCL(cls_feature, labels,method='eucilidean')

                    loss = loss_id + loss_tri + cfg.MA * loss_ma+ cfg.GMSEL * loss_gmsel+ cfg.GDCL * loss_gdcl
                    #
                    #
                    #

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_rgb = (score_color.max(1)[1] == label_color).float().mean()
            acc_ir = (score_thermal.max(1)[1] == label_thermal).float().mean()

            loss_tri_meter.update(loss_tri.item())
            loss_ce_meter.update(loss_id.item())
            loss_meter.update(loss.item())

            acc_rgb_meter.update(acc_rgb, 1)
            acc_ir_meter.update(acc_ir, 1)

            torch.cuda.synchronize()

            if (idx + 1) % 64 == 0:
                print('Epoch[{}] Iteration[{}/{}]'
                      ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                      'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                      'Base Lr: {:.2e} '.format(epoch, (idx + 1),
                                                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                                                loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                                                optimizer.state_dict()['param_groups'][0]['lr']))
    elif cfg.METHOD == 'TripletModality_colorEmbedding2':
        for idx, (input_color, input_thermal, input_gcolor, input_gthermal,
                  label_color, label_thermal, label_gcolor, label_gthermal) in enumerate(tqdm(trainloader)):
            # img_color, img_gray,img_thermal, target_color, target_gray, target_color
            optimizer.zero_grad()
            input_color = input_color.to(device)
            input_gcolor = input_gcolor.to(device)
            input_gthermal = input_gthermal.to(device)
            input_thermal = input_thermal.to(device)

            label_color = label_color.to(device)
            label_gcolor = label_gcolor.to(device)
            label_gthermal = label_gthermal.to(device)
            label_thermal = label_thermal.to(device)

            # cutmix_prob = np.random.rand(1)
            # if cutmix_prob < 0.2:
            #     lam = np.random.beta(1, 1)
            #     _,_,H,W=input_color.size()
            #     bbx1, bby1, bbx2, bby2 = rand_bbox(H, W, lam)
            #     input_color[:,:,bbx1:bbx2, bby1:bby2],input_gcolor[:,:,bbx1:bbx2, bby1:bby2]=\
            #         input_gcolor[:,:,bbx1:bbx2, bby1:bby2],input_color[:,:,bbx1:bbx2, bby1:bby2]
            #     input_thermal[:, :, bbx1:bbx2, bby1:bby2], input_gthermal[:, :, bbx1:bbx2, bby1:bby2] = \
            #         input_gthermal[:, :, bbx1:bbx2, bby1:bby2], input_thermal[:, :, bbx1:bbx2, bby1:bby2]

            labels = torch.cat((label_color, label_gcolor, label_gthermal, label_thermal), 0)

            with amp.autocast(enabled=True):
                # color_diff=input_color-input_gray

                cls_score, cls_feature, cls_feature_bn, g_feature = model(
                    torch.cat([input_color, input_gcolor, input_gthermal, input_thermal]))

                # loss =MA_eucilidean(cls_score,cls_feature,label_color,label_gray,label_thermal,labels,g_feature)

                score_color, score_gcolor, score_gthermal, score_thermal = cls_score.chunk(4, 0)
                feat_color, feat_gcolor, feat_gthermal, feat_thermal = cls_feature.chunk(4, 0)

                loss_id = criterion_ID(score_color, label_color.long()) + \
                          criterion_ID(score_gcolor, label_gcolor.long()) + \
                          criterion_ID(score_gthermal, label_gthermal.long()) + \
                          criterion_ID(score_thermal, label_thermal.long())
                # loss_tri_intra=0.5*criterion_Intra(torch.cat([feat_color,feat_gcolor],dim=0),
                #                              torch.cat([feat_color, feat_gcolor], dim=0))+ \
                #                0.5*criterion_Intra(torch.cat([feat_thermal, feat_gthermal],dim=0),
                #                              torch.cat([feat_thermal, feat_gthermal], dim=0))
                # loss_tri_intra=criterion_Tri(torch.cat([feat_color,feat_gcolor],dim=0),
                #                              torch.cat([feat_color,feat_gcolor],dim=0),
                #                              torch.cat([label_color, label_gcolor], dim=0),
                #                              )+ \
                #                criterion_Tri(torch.cat([feat_thermal, feat_gthermal],dim=0),
                #                              torch.cat([feat_thermal, feat_gthermal],dim=0),
                #                              torch.cat([label_gthermal, label_thermal], dim=0),
                #                              )
                # loss_ma = criterion_MA(cls_feature, g_feature, labels, method='eucilidean')
                loss_ma = criterion_MA(cls_feature, g_feature, labels, method='cosine')
                # loss_pm=criterion_PL(patch_features)

                if epoch <= cfg.PL_EPOCH:
                    loss_tri = criterion_Tri(feat_color, feat_color, label_color) + \
                               criterion_Tri(feat_gcolor, feat_gcolor, label_gcolor) + \
                               criterion_Tri(feat_gthermal, feat_gthermal, label_gthermal) + \
                               criterion_Tri(feat_thermal, feat_thermal, label_thermal)
                    # loss_pth = criterion_PL(cls_feature=cls_feature,patch_features=patch_features)

                    # loss = loss_id + loss_tri + cfg.MA * loss_ma

                    loss = loss_id + loss_tri + cfg.MA * loss_ma

                else:
                    loss_tri = criterion_Tri(cls_feature, cls_feature, labels)

                    loss_gmsel = criterion_GMSEL(cls_feature, labels)
                    loss_gdcl = criterion_GCL(cls_feature, labels)

                    # loss = loss_id + loss_tri + cfg.GDCL * loss_gdcl + cfg.GMSEL * loss_gmsel+ cfg.MA*loss_ma #+cfg.PM*loss_pm
                    # loss = loss_id + loss_tri + cfg.MA * loss_ma + cfg.GMSEL * loss_gmsel#+ cfg.GDCL * loss_gdcl
                    loss = loss_id + loss_tri + cfg.MA * loss_ma + cfg.GMSEL * loss_gmsel + cfg.GDCL * loss_gdcl  #
                    # cfg.MA * loss_ma  +cfg.GMSEL * loss_gmsel+ cfg.GDCL * loss_gdcl
                    # loss = loss_id + loss_tri + cfg.MA * loss_ma
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc_rgb = (score_color.max(1)[1] == label_color).float().mean()
            acc_ir = (score_thermal.max(1)[1] == label_thermal).float().mean()

            loss_tri_meter.update(loss_tri.item())
            loss_ce_meter.update(loss_id.item())
            loss_meter.update(loss.item())

            acc_rgb_meter.update(acc_rgb, 1)
            acc_ir_meter.update(acc_ir, 1)

            torch.cuda.synchronize()

            if (idx + 1) % 64 == 0:
                print('Epoch[{}] Iteration[{}/{}]'
                      ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                      'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                      'Base Lr: {:.2e} '.format(epoch, (idx + 1),
                                                len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                                                loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                                                optimizer.state_dict()['param_groups'][0]['lr']))
    else:
        assert 1 < 0, cfg.METHOD + 'is not existing'

    end_time = time.time()
    time_per_batch = end_time - start_time
    print(' Epoch {} done. Time per batch: {:.1f}[min] '.format(epoch, time_per_batch / 60))


def test(query_loader, gall_loader, dataset='sysu'):
    model.eval()
    nquery = len(query_label)
    ngall = len(gall_label)
    print('Testing...')
    t = time.localtime()
    print(time.asctime(t))

    ptr = 0
    gall_feat = np.zeros((ngall, 768))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(tqdm(gall_loader)):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    ptr = 0
    query_feat = np.zeros((nquery, 768))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(tqdm(query_loader)):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat = model(input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

    distmat = -np.matmul(query_feat, np.transpose(gall_feat))
    if dataset == 'sysu':
        cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
    else:
        cmc, mAP, mInp = eval_regdb(distmat, query_label, gall_label)

    return cmc, mAP, mInp


if __name__ == '__main__':
    # Training
    best_mAP = 0
    print('==> Start Training...')
    t = time.localtime()
    print(time.asctime(t))

    for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH * 3):

        print('==> Preparing Data Loader...')
        t = time.localtime()
        print(time.asctime(t))

        if cfg.METHOD == 'VI_two_Modality':
            sampler = TwoMSampler(trainset.train_color_label,
                                  trainset.train_thermal_label,
                                  color_pos_rgb, thermal_pos_rgb, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)
            trainset.cIndex = sampler.index_color
            trainset.tIndex = sampler.index_thermal
            trainloader = data.DataLoader(trainset, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
        elif cfg.METHOD == 'RGB_gary_Modality' or cfg.METHOD == 'IR_gary_Modality':
            sampler = Tri1Sampler(trainset.train_color_label,
                                  trainset.train_thermal_label,
                                  color_pos_rgb, thermal_pos_rgb, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)
            trainset.cIndex = sampler.index_color
            trainset.gIndex = sampler.index_gray
            trainset.tIndex = sampler.index_thermal
            trainloader = data.DataLoader(trainset, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                                          num_workers=args.num_workers, drop_last=True, pin_memory=True)

        else:
            sampler_tri = Tri2Sampler(trainset.train_color_label,
                                      trainset.train_thermal_label,
                                      color_pos_rgb, thermal_pos_rgb, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)
            trainset.cIndex = sampler_tri.index_color
            # trainset.gIndex = sampler_tri.index_gray
            trainset.gcIndex = sampler_tri.index_gcolor
            trainset.gtIndex = sampler_tri.index_gthermal
            trainset.tIndex = sampler_tri.index_thermal
            trainloader = data.DataLoader(trainset, batch_size=cfg.BATCH_SIZE, sampler=sampler_tri,
                                          num_workers=args.num_workers, drop_last=True, pin_memory=True)

        train(epoch)

        if epoch > args.start_test and epoch % args.test_epoch == 0:
            cmc, mAP, mINP = test(query_loader, gall_loader, cfg.DATASET)
            result_statistics.add_row([epoch,
                                       '{:.2%}'.format(mAP),
                                       '{:.2%}'.format(mINP),
                                       '{:.2%}'.format(cmc[0]),
                                       '{:.2%}'.format(cmc[4]),
                                       '{:.2%}'.format(cmc[9]),
                                       '{:.2%}'.format(cmc[19])]
                                      )
            if epoch >= 30 and epoch % 10 == 0:
                result_statistics.sortby = 'Rank-1'
                logger(str(result_statistics))
            else:
                print(str(result_statistics))

            if mAP > best_mAP:
                best_mAP = mAP
                if cfg.DATASET == 'sysu':
                    torch.save(model.state_dict(),
                               osp.join(cfg.logger_dir, os.path.basename(args.config_file)[:-4] + '_best.pth')
                               )  # maybe not the best
                else:
                    torch.save(model.state_dict(),
                               osp.join(cfg.logger_dir,
                                        os.path.basename(args.config_file)[:-4] + '_best.pth'.format(
                                            args.trial))
                               )

        if epoch > 20 and epoch % args.save_epoch == 0:
            torch.save(model.state_dict(),
                       osp.join(cfg.logger_dir, os.path.basename(args.config_file)[:-4] + '_epoch{}.pth'.format(epoch)))
