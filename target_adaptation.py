import argparse
import os, sys
import os.path as osp
import time

import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from numpy import linalg as LA
from torch.nn import functional as F
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.manifold import TSNE
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from tqdm import tqdm

def get_posAndneg(args, features, labels, tgt_label=None, feature_q_idx=None, co_fea=None):
    # get the label of q
    q_label = tgt_label[feature_q_idx]

    # get the positive sample
    positive_sample_idx = []
    for i, label in enumerate(labels):
        if label == q_label:
            positive_sample_idx.append(i)

    if len(positive_sample_idx) != 0:
        feature_pos = features[random.choice(positive_sample_idx)].unsqueeze(0)
    else:
        feature_pos = co_fea.unsqueeze(0)

    # get the negative samples
    negative_sample_idx = []
    for idx in range(features.shape[0]):
        if labels[idx] != q_label:
            negative_sample_idx.append(idx)

    negative_pairs = torch.Tensor([]).cuda()
    ii = 0
    while ii < args.class_num :
        cur_choice = features[random.choice(negative_sample_idx)]
        if cur_choice.tolist() == (torch.zeros(256).cuda()).tolist():
            continue
        negative_pairs = torch.cat((negative_pairs, features[random.choice(negative_sample_idx)].unsqueeze(0)))
        ii = ii + 1
    # if negative_pairs.shape[0] == args.class_num - 1:
    #     features_neg = negative_pairs
    # else:
    #     raise Exception('Negative samples error!')
    features_neg = negative_pairs
    return torch.cat((feature_pos, features_neg))


def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())
    return similarity


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()



    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label_ts(loader, netF, netB, netC, netC_RE, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas_F = netF(inputs)
            feas = netB(feas_F)
            envidence = netC_RE(feas)

            outputs = netC(feas)
            if start_test:
                all_fea_F = feas_F.float().cpu()
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_envidence = envidence.float().cpu()

                all_label = labels.float()
                start_test = False
            else:
                all_fea_F = torch.cat((all_fea_F, feas_F.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_envidence = torch.cat((all_envidence, envidence.float().cpu()), 0)

                all_label = torch.cat((all_label, labels.float()), 0)

    evidence = F.relu(all_envidence)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    u = args.class_num / S
    u = u.squeeze(-1)
    output_norm = F.normalize(all_fea)
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    len_unconfi = int(u.shape[0] * 0.5)  # unconfident group length
    idx_unconfi = u.topk(len_unconfi, largest=False)[-1]
    idx_unconfi_list_evidence = idx_unconfi.cpu().numpy().tolist()

    len_unconfi = int(ent.shape[0] * 0.5)  # unconfident group length
    idx_unconfi = ent.topk(len_unconfi, largest=False)[-1]
    idx_unconfi_list_ent = idx_unconfi.cpu().numpy().tolist()

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    # --------------------use dd to get confi_idx and unconfi_idx-------------
    dd_min = dd.min(axis=1)
    dd_min_tsr = torch.from_numpy(dd_min).detach()
    dd_t_confi = dd_min_tsr.topk(int((dd.shape[0] * 0.5)), largest=False)[-1]
    dd_confi_list = dd_t_confi.cpu().numpy().tolist()
    dd_confi_list.sort()
    idx_confi = dd_confi_list

    idx_all_arr = np.zeros(shape=dd.shape[0], dtype=np.int64)
    # idx_all_list = list(idx_all_arr)
    idx_all_arr[idx_confi] = 1
    idx_unconfi_arr = np.where(idx_all_arr == 1)
    idx_unconfi_list_dd = list(idx_unconfi_arr[0])

    if args.uncertainty == 'dd_en_ev':
        idx_unconfi_list = list(set(idx_unconfi_list_dd).intersection(set(idx_unconfi_list_ent)).intersection(
            set(idx_unconfi_list_evidence)))
    elif args.uncertainty == 'dd_en':
        idx_unconfi_list = list(set(idx_unconfi_list_dd).intersection(set(idx_unconfi_list_ent)))
    elif args.uncertainty == 'dd_ev':
        idx_unconfi_list = list(set(idx_unconfi_list_dd).intersection(set(idx_unconfi_list_evidence)))
    elif args.uncertainty == 'en_ev':
        idx_unconfi_list = list(set(idx_unconfi_list_evidence).intersection(set(idx_unconfi_list_ent)))
    elif args.uncertainty == 'dd':
        idx_unconfi_list = list(set(idx_unconfi_list_dd))
    elif args.uncertainty == 'en':
        idx_unconfi_list = list(set(idx_unconfi_list_ent))
    elif args.uncertainty == 'ev':

        idx_unconfi_list = list(set(idx_unconfi_list_evidence))




    # ------------------------------------------------------------------------
    label_confi = np.zeros(ent.shape[0], dtype="int64")
    if args.uncertainty != 'null':
        label_confi[idx_unconfi_list] = 1

    _, all_idx_nn, _ = obtain_nearest_trace(all_fea_F, all_fea_F, label_confi)

    ln = label_confi.shape[0]
    gamma = 0.15 * np.random.randn(ln, 1) + 0.85

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd_sf = cdist(all_fea, initc[labelset], args.distance)
        dd_nn = dd_sf[all_idx_nn]
        dd = gamma * dd_sf + (1 - gamma) * dd_nn
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)


    # calculate the confidence weights
    max_prob, _ = torch.max(F.softmax(torch.from_numpy(1 - dd) / 0.07, dim=1), dim=1)

    # calculate the centord
    confid_label = (label_confi * 2 - 1) * (pred_label + 1)
    start_test = True
    centord_lab = []
    for i in range(args.class_num + 1):
        if i == 0:
            continue
        same_ind = np.where(confid_label == i)
        find_ind = same_ind[0]
        ii = 0
        if find_ind.shape[0] == 0:
            same_ind = np.where(pred_label == (i - 1))
            find_ind = same_ind[0]
        if find_ind.shape[0] == 0:  ## find_ind = np.array(find_ind)
            cur_centord = torch.zeros(2,256)
            centord_lab.append(i - 1)
            centord_lab.append(i - 1)
            if start_test:
                centord_fea = cur_centord
                start_test = False
            else:
                centord_fea = torch.cat((centord_fea, cur_centord), 0)
            continue
        while find_ind.shape[0] < 2:
            find_ind = np.append(find_ind, find_ind[ii])
            ++ ii
        cur_centord = output_norm[find_ind[max_prob[find_ind].topk(2, largest=True)[-1]]]
        centord_lab.append(i - 1)
        centord_lab.append(i - 1)
        if start_test:
            centord_fea = cur_centord
            start_test = False
        else:
            centord_fea = torch.cat((centord_fea, cur_centord), 0)

    return pred_label.astype(
        'int'), all_fea_F, label_confi, all_label, output_norm, all_output.cuda(), max_prob.detach(), centord_fea.cuda(), np.array(
        centord_lab)


def obtain_nearest_trace(data_q, data_all, lab_confi):
    data_q_ = data_q.detach()
    data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()
    num_sam = data_q.shape[0]
    LN_MEM = 70

    flag_is_done = 0  # indicate whether the trace process has done over the target dataset
    ctr_oper = 0  # counter the operation time
    idx_left = np.arange(0, num_sam, 1)
    mtx_mem_rlt = -3 * np.ones((num_sam, LN_MEM), dtype='int64')
    mtx_mem_ignore = np.zeros((num_sam, LN_MEM), dtype='int64')
    is_mem = 0
    mtx_log = np.zeros((num_sam, LN_MEM), dtype='int64')
    indices_row = np.arange(0, num_sam, 1)
    flag_sw_bad = 0
    nearest_idx_last = np.array([-7])

    while flag_is_done == 0:

        nearest_idx_tmp, idx_last_tmp = get_nearest_sam_idx(data_q_, data_all_, is_mem, ctr_oper, mtx_mem_ignore,
                                                            nearest_idx_last)
        is_mem = 1
        nearest_idx_last = nearest_idx_tmp

        if ctr_oper == (LN_MEM - 1):
            flag_sw_bad = 1
        else:
            flag_sw_bad = 0

        mtx_mem_rlt[:, ctr_oper] = nearest_idx_tmp
        mtx_mem_ignore[:, ctr_oper] = idx_last_tmp

        lab_confi_tmp = lab_confi[nearest_idx_tmp]
        idx_done_tmp = np.where(lab_confi_tmp == 1)[0]
        idx_left[idx_done_tmp] = -1

        if flag_sw_bad == 1:
            idx_bad = np.where(idx_left >= 0)[0]
            mtx_log[idx_bad, 0] = 1
        else:
            mtx_log[:, ctr_oper] = lab_confi_tmp

        flag_len = len(np.where(idx_left >= 0)[0])

        if flag_len == 0 or flag_sw_bad == 1:
            idx_nn_step = []
            for k in range(num_sam):
                try:
                    idx_ts = list(mtx_log[k, :]).index(1)
                    idx_nn_step.append(idx_ts)
                except:
                    idx_nn_step.append(0)

            idx_nn_re = mtx_mem_rlt[indices_row, idx_nn_step]
            data_re = data_all[idx_nn_re, :]
            flag_is_done = 1
        else:
            data_q_ = data_all_[nearest_idx_tmp, :]
        ctr_oper += 1

    return data_re, idx_nn_re, idx_nn_step


def get_nearest_sam_idx(Q, X, is_mem_f, step_num, mtx_ignore,
                        nearest_idx_last_f):  # Q、X arranged in format of row-vector
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    Sim = 1 - (Simo / Nor)

    indices_min = np.argmin(Sim, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)

    idx_change = np.where((indices_min - nearest_idx_last_f) != 0)[0]
    if is_mem_f == 1:
        if idx_change.shape[0] != 0:
            indices_min[idx_change] = nearest_idx_last_f[idx_change]
    Sim[indices_row, indices_min] = 1000

    # Ignore the history elements.
    if is_mem_f == 1:
        for k in range(step_num):
            indices_ingore = mtx_ignore[:, k]
            Sim[indices_row, indices_ingore] = 1000

    indices_min_cur = np.argmin(Sim, axis=1)
    indices_self = indices_min
    return indices_min_cur, indices_self


def train_target(args):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger = SummaryWriter(args.output_dir)
    dset_loaders = data_load(args)

    # set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC_RE = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C_RE.pt'
    netC_RE.load_state_dict(torch.load(modelpath))

    netC.eval()
    netC_RE.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netC_RE.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    best_acc = 0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()

            mem_label, feas_all, label_confi, label_all, fea_bank, score_bank, confidence_weight, centord_img, centord_lab \
                = obtain_label_ts(dset_loaders['test'], netF, netB, netC, netC_RE, args)
            mem_label = torch.from_numpy(mem_label).cuda()

            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test_F = netF(inputs_test)
        features_test_N, idx_nn_re, _ = obtain_nearest_trace(features_test_F, feas_all, label_confi)
        idx_nn_re_final = np.expand_dims(idx_nn_re,axis=1)
        if args.K > 1:
            features_test_tofind = features_test_N
            for ii in range(args.K - 1):
                features_test_tofind, idx_nn_re_cur, _ = obtain_nearest_trace(features_test_tofind, feas_all, label_confi)
                idx_nn_re_cur = np.expand_dims(idx_nn_re_cur,axis=1)
                idx_nn_re_final = np.concatenate((idx_nn_re_final, idx_nn_re_cur), axis=1)


        features_test_F = netB(features_test_F)
        outputs_test_F = netC(features_test_F)

        features_test_N = features_test_N.cuda()
        features_test_N = netB(features_test_N)
        outputs_test_N = netC(features_test_N)

        pred = mem_label[tar_idx]

        if args.envi_par > 0:
            outputs_evidence_F = netC_RE(features_test_F)
            outputs_evidence_N = netC_RE(features_test_N)
            y = one_hot_embedding(pred.long(), args.class_num).cuda()
            cevidence_loss_F = loss.edl_mse_loss(outputs_evidence_F, y.float(), iter_num, args.class_num, 10, device)
            cevidence_loss_N = loss.edl_mse_loss(outputs_evidence_N, y.float(), iter_num, args.class_num, 10, device)
            cevidence_loss = 1.0 * cevidence_loss_F + 1.0 * cevidence_loss_N
            logger.add_scalar('cevidence_loss', cevidence_loss.item(), iter_num)
        else:
            cevidence_loss = torch.tensor(0.0).cuda()

        if args.cls_par > 0:
            classifier_loss_F = nn.CrossEntropyLoss()(outputs_test_F, pred.long())
            classifier_loss_N = nn.CrossEntropyLoss()(outputs_test_N, pred.long())
            classifier_loss = 1.0 * classifier_loss_F + 1.0 * classifier_loss_N
            logger.add_scalar('classifier_loss', classifier_loss.item(), iter_num)

        else:
            classifier_loss = torch.tensor(0.0).cuda()


        if args.ent:
            softmax_out_F = nn.Softmax(dim=1)(outputs_test_F)
            softmax_out_N = nn.Softmax(dim=1)(outputs_test_N)
            softmax_out = 1.0 * softmax_out_F + 1.0 * softmax_out_N

            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            logger.add_scalar('entropy_loss', entropy_loss.item(), iter_num)

        else:
            entropy_loss = torch.tensor(0.0).cuda()

        # NNH
        outputs_softmax_out_F = nn.Softmax(dim=1)(outputs_test_F)
        with torch.no_grad():
            output_f_norm = F.normalize(features_test_F)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = outputs_softmax_out_F.detach().clone()
            confid_idx_near_exclude = torch.from_numpy(idx_nn_re_final)
            confid_score_near = score_bank[confid_idx_near_exclude]
            weight = torch.ones([confid_score_near.shape[0], args.K])

        softmax_out_un = outputs_softmax_out_F.unsqueeze(1).expand(-1, args.K, -1)


        loss_N = torch.mean(
            (F.kl_div(softmax_out_un, confid_score_near, reduction='none').sum(-1) *
             weight.cuda()).sum(1))
        logger.add_scalar('loss_N', loss_N.item(), iter_num)

        mean_softmax = outputs_softmax_out_F.mean(dim=0)
        im_div = torch.sum(mean_softmax * torch.log(mean_softmax + 1e-5))
        # logger.add_scalar('im_div', im_div.item(), iter_num)

        loss_const = -torch.mean(outputs_softmax_out_F @ score_bank.T)
        # logger.add_scalar('loss_const', loss_const.item(), iter_num)


        all_sam_indx, all_in, _ = np.intersect1d(tar_idx.numpy(), tar_idx.numpy(), return_indices=True)
        total_contrastive_loss = Variable(torch.tensor(0.).cuda())
        contrastive_label = torch.tensor([0]).cuda()
        gamma = 0.07  # temperature
        nll = nn.NLLLoss()
        if len(all_in) > 0:
            for idx in range(len(all_in)):
                pairs4q = get_posAndneg(args=args, features=centord_img,
                                        labels=torch.from_numpy(centord_lab).cuda(),
                                        tgt_label=mem_label.cpu().numpy(),
                                        feature_q_idx=tar_idx.numpy()[all_in[idx]],
                                        co_fea=features_test_F[
                                            all_in[idx]].cuda())

                # calculate cosine similarity [-1 1]
                result = cosine_similarity(features_test_F[all_in[idx]].unsqueeze(0).cuda(), pairs4q)

                numerator = 1

                denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                # log
                result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)  #
                # nll_loss
                contrastive_loss = nll(result, contrastive_label) * confidence_weight[tar_idx[all_in[idx]]]
                total_contrastive_loss = total_contrastive_loss + contrastive_loss
            total_contrastive_loss = total_contrastive_loss / len(all_in)
        logger.add_scalar('total_contrastive_loss', total_contrastive_loss.item(), iter_num)

        # lamda = (2 / (1 + math.exp(-10 * (iter_num) / (max_iter))) - 1)*0.8+0.2
        all_loss = args.pos_cl_par * loss_N + im_div + loss_const + \
                    args.envi_par * cevidence_loss + \
                    args.cls_par * classifier_loss + \
                    args.cl_par * total_contrastive_loss + \
                    args.ent_par * entropy_loss
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        loginterval =max_iter // 100
        if loginterval == 0:
            loginterval = 1
        if iter_num % loginterval == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()

            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str)
            args.out_file.flush()
            if best_acc <= acc_s_te:
                best_acc = acc_s_te
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_best" + ".pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_best" + ".pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_best" + ".pt"))
            else:
                torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_cur" + ".pt"))
                torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_cur" + ".pt"))
                torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_cur" + ".pt"))
            netF.train()
            netB.train()
            args.out_file.write(" bestAccuracy = " + str(best_acc)[0:6] + '%\n')
            args.out_file.flush()
            print(log_str + " bestAccuracy = " + str(best_acc)[0:6] + '%\n')
            logger.add_scalar('acc', acc_s_te, iter_num)
            logger.add_scalar('total_contrastive_loss', total_contrastive_loss.item(), iter_num)
            logger.add_scalar('cevidence_loss', cevidence_loss.item(), iter_num)
            logger.add_scalar('loss_N', loss_N.item(), iter_num)
            logger.add_scalar('entropy_loss', entropy_loss.item(), iter_num)
            logger.add_scalar('classifier_loss', classifier_loss.item(), iter_num)

    return netF, netB, netC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UC-SFDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office',
                        choices=['office', 'office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='vit', help="resnet50, vit")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--uncertainty', type=str, default='dd_en_ev', help="dd_en_ev, dd_en, dd_ev, en_ev, dd, en, ev, null")
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)

    parser.add_argument('--threshold', type=int, default=0)

    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--envi_par', type=float, default=0.1)
    parser.add_argument('--cl_par', type=float, default=0.2)
    parser.add_argument('--pos_cl_par', type=float, default=1)

    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])

    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--output', type=str, default='source_ecl_transformer')
    parser.add_argument('--file', type=str, default='run1')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    args.output_dir_src = osp.join(args.output, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.dset, names[args.s][0].upper(),
                               names[args.s][0].upper() + names[args.t][0].upper(), args.file)
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.out_file = open(
        osp.join(args.output_dir, str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))) + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    train_target(args)
    event_data = event_accumulator.EventAccumulator(args.output_dir)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys):
        # print(key)
        if key != 'train/total_loss_iter':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
            df[key] = pd.DataFrame(event_data.Scalars(key)).value

    df.to_csv(args.output_dir+'\\new.csv')
