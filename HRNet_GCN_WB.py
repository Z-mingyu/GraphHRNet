from __future__ import print_function, absolute_import, division

import random
import os
import time
import datetime
import argparse
import numpy as np
import shutil
import json
import os.path as path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from lib.config import cfg
from einops import rearrange, repeat

from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.graph_utils import adj_mx_from_skeleton
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe, sym_penalty
from common.camera import get_uvd2xyz
from utils.prepare_data_h3wb import Human3WBDataset, TRAIN_SUBJECTS, TEST_SUBJECTS

from models.graph_sh import GraphSH
import models.graph_hrnet_multi_branch as ghrmb
import models.graph_resnet as GraphRes
import models.graph_hrnet as ghr
import models.graph_resnet_four_branch as GraphRes_4


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h3wb_', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-cfg', '--configuration', default='w32_adam_lr1e-3.yaml', type=str, metavar='NAME',
                        help='configuration file')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=50, type=int, help='save models for every #snapshot epochs (default: 20)')
    parser.add_argument('-m', '--model', default=1, type=int, help='index of model(1,2,3,4)')

    # Model arguments
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('-z', '--hid_dim', default=64, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--lamda', '--weight_L1_norm', default=0, type=float, metavar='N', help='scale of L1 Norm')
    parser.add_argument('--num_workers', default=24, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-2, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=500, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.99, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--post_refine', dest='post_refine', action='store_true', help='if use post-refine layers')
    parser.set_defaults(post_refine=False)
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate')
    parser.add_argument('--gcn', default='dc_preagg', type=str, metavar='NAME', help='type of gcn')
    parser.add_argument('-n', '--name', default='', type=str, metavar='NAME', help='name of model')

    # Experimental
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# S1, S5, S6 and S7, including 80k {image,2D,3D} triplets. The test set contains all samples from S8, including 20k triplets.
def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', args.dataset + 'train.npz')
    dataset_test_path = path.join('data', args.dataset + 'test.npz')
    dataset = Human3WBDataset(dataset_path, dataset_test_path)
    kps = dataset.keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(kps[0]), list(kps[1])
    subjects_train = TRAIN_SUBJECTS
    subjects_test = TEST_SUBJECTS

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')

    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda:0")

    # Create model
    print("==> Creating model...")

    p_dropout = (None if args.dropout == 0.0 else args.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton()).to(device)

    model_post_refine = None
    cfg.merge_from_file(args.configuration)
    if args.model == 1:
        model_pos = ghrmb.get_pose_net(cfg, True, adj, p_dropout, args.gcn, dataset.skeleton().joints_group()).to(
            device)
    elif args.model == 2:

        model_pos = GraphRes.get_pose_net(True, adj, p_dropout, args.gcn, 50, True).to(device)
    elif args.model == 3:
        model_pos = ghr.get_pose_net(cfg, True, adj, p_dropout, args.gcn, dataset.skeleton().joints_group()).to(device)
    elif args.model == 4:
        model_pos = GraphSH(adj, args.hid_dim, dataset.skeleton().joints_group(), num_layers=args.num_layers,
                            p_dropout=p_dropout, gcn_type=args.gcn).to(device)
    elif args.model == 5:

        model_pos = GraphRes_4.get_pose_net(True, adj, p_dropout, args.gcn, 50, True).to(device)
    else:
        raise TypeError("index of model doesn't exist")

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').to(device)
    criterionL1 = nn.L1Loss(reduction='mean').to(device)

    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
            # for name, p in model_pos.named_parameters():
            #    print(name)#, p.data)
            # exit(0)

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        if args.model == 1:

            ckpt_dir_path = path.join(args.checkpoint, "HRGCN",
                                      args.gcn + '-' + datetime.datetime.now().replace(microsecond=0).isoformat())
        elif args.model == 2:
            ckpt_dir_path = path.join(args.checkpoint, 'GraphRes',
                                      args.gcn + '-' + datetime.datetime.now().replace(microsecond=0).isoformat())
        elif args.model == 3:

            ckpt_dir_path = path.join(args.checkpoint, 'HRGCN*',
                                      args.gcn + '-' + datetime.datetime.now().replace(microsecond=0).isoformat())
        elif args.model == 4:
            ckpt_dir_path = path.join(args.checkpoint, 'GraphSH',
                                      args.gcn + '-' + datetime.datetime.now().replace(microsecond=0).isoformat())
        elif args.model == 5:
            ckpt_dir_path = path.join(args.checkpoint, 'GraphRes_4',
                                      args.gcn + '-' + datetime.datetime.now().replace(microsecond=0).isoformat())
        else:
            raise TypeError("index of model doesn't exist")

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_body', 'error_face',
                          'error_hand', 'error_face_aligned', 'error_hand_aligned'])

        logger_action = Logger(os.path.join(ckpt_dir_path, 'action.txt'))
        logger_action.set_names(
            ['epoch', 'Dir.', 'Disc.', 'Eat', 'Greet', 'Phone', 'Photo', 'Pose', 'Pur.', 'Sit', 'SitD.', 'Smoke',
             'Wait', 'WalkD.', 'Walk', 'WalkT', 'Avg'])
        shutil.copy(args.configuration, ckpt_dir_path)

        with open(os.path.join(ckpt_dir_path, "params.json"), mode="w") as f:
            json.dump(args.__dict__, f, indent=4)

    if args.evaluate:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        error_eval_p1 = np.zeros(len(action_filter))
        error_body = np.zeros(len(action_filter))
        error_face = np.zeros(len(action_filter))
        error_hand = np.zeros(len(action_filter))
        error_face_aligned = np.zeros(len(action_filter))
        error_hand_aligned = np.zeros(len(action_filter))

        print(action_filter)
        for i, action in enumerate(action_filter):
            print(i, ' ', action)
            poses_valid, poses_valid_2d, actions_valid, cam_valid = fetch(subjects_test, dataset, keypoints, [action],
                                                                          stride)

            if poses_valid is not None:
                valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid, cam_valid),
                                          batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)

                error_eval_p1[i], error_body[i], error_face[i], error_hand[i], error_face_aligned[i], \
                    error_hand_aligned[
                        i] = \
                    evaluate(valid_loader, model_pos, device, kps_left, kps_right)

        # print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(error_eval_p1).item()))
        # print('Protocol #1   (MPJPE) action-wise average-body: {:.2f} (mm)'.format(np.mean(error_body).item()))
        # print('Protocol #1   (MPJPE) action-wise average-face: {:.2f} (mm)'.format(np.mean(error_face).item()))
        # print('Protocol #1   (MPJPE) action-wise average-hand: {:.2f} (mm)'.format(np.mean(error_hand).item()))
        # print('Protocol #1   (MPJPE) action-wise average-face-aligned: {:.2f} (mm)'.format(
        #     np.mean(error_face_aligned).item()))
        # print('Protocol #1   (MPJPE) action-wise average-hand-aligned: {:.2f} (mm)'.format(
        #     np.mean(error_hand_aligned).item()))
        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.sum(error_eval_p1) / 12))
        print('Protocol #1   (MPJPE) action-wise average-body: {:.2f} (mm)'.format(np.sum(error_body) / 12))
        print('Protocol #1   (MPJPE) action-wise average-face: {:.2f} (mm)'.format(np.sum(error_face) / 12))
        print('Protocol #1   (MPJPE) action-wise average-hand: {:.2f} (mm)'.format(np.sum(error_hand) / 12))
        print('Protocol #1   (MPJPE) action-wise average-face-aligned: {:.2f} (mm)'.format(
            np.sum(error_face_aligned) / 12))
        print('Protocol #1   (MPJPE) action-wise average-hand-aligned: {:.2f} (mm)'.format(
            np.sum(error_hand_aligned) / 12))
        exit(0)

    poses_train, poses_train_2d, actions_train, cam_train = fetch(subjects_train, dataset, keypoints, action_filter,
                                                                  stride)
    poses_valid, poses_valid_2d, actions_valid, cam_valid = fetch(subjects_test, dataset, keypoints, action_filter,
                                                                  stride)

    train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, actions_train, cam_train),
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid, cam_valid),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        # print(len(train_loader))
        epoch_loss, lr_now, glob_step = train(train_loader, model_pos, model_post_refine,
                                              args.lamda, criterion, criterionL1, optimizer,
                                              device, args.lr, lr_now,
                                              glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)

        # Evaluate
        if action_filter is None:
            action_filter = dataset.define_actions()
        error_eval_p1, error_body, error_face, error_hand, error_face_aligned, error_hand_aligned = \
            evaluate(valid_loader, model_pos, device, kps_left, kps_right)
        # error=evaluate_action(model_pos, device, action_filter, dataset, keypoints,kps_left,kps_right)

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_body, error_face, error_hand,
                       error_face_aligned, error_hand_aligned])

        # logger_action.append([epoch + 1, error[0], error[1], error[2], error[3], error[4], error[5], error[6], error[7], error[8],
        #                error[9], error[10], error[11], error[12], error[13], error[14],np.sum(error)/12])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'post_refine': model_post_refine, 'error': error_eval_p1},
                      ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'post_refine': model_post_refine, 'error': error_eval_p1},
                      ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))

    return


def train(data_loader, model_pos, model_post_refine, lamda, criterion, criterionL1, optimizer, device, lr_init, lr_now,
          step, decay, gamma,
          max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    if model_post_refine is not None:
        model_post_refine.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, batch_cam) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d, batch_cam = targets_3d.to(device), inputs_2d.to(device), batch_cam.to(device)
        # body_3d, face_3d, hand_3d = model_pos(inputs_2d)
        body_3d, face_3d, left_hand_3d, right_hand_3d = model_pos(inputs_2d)

        loss_post_refine = 0

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)

        loss_3d_body = ((1 - lamda) * criterion(body_3d, targets_3d[:, :23, :]) + lamda *
                        criterionL1(body_3d, targets_3d[:, :23, :]) + loss_post_refine)

        loss_3d_face = ((1 - lamda) * criterion(face_3d, targets_3d[:, 23:91, :]) + lamda *
                        criterionL1(face_3d, targets_3d[:, 23:91, :]) + loss_post_refine)

        loss_3d_left_hand = ((1 - lamda) * criterion(left_hand_3d, targets_3d[:, 91:112, :]) + lamda *
                             criterionL1(left_hand_3d, targets_3d[:, 91:112, :]) + loss_post_refine)

        loss_3d_right_hand = ((1 - lamda) * criterion(right_hand_3d, targets_3d[:, 112:, :]) + lamda *
                              criterionL1(right_hand_3d, targets_3d[:, 112:, :]) + loss_post_refine)

        loss_3d_pos = 2*loss_3d_body + 2*loss_3d_face + loss_3d_left_hand + loss_3d_right_hand
        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .6f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


def evaluate_action(model_pos, device, action_filter, dataset, keypoints, kps_left, kps_right):
    error_eval_p1 = np.zeros(len(action_filter))
    error_body = np.zeros(len(action_filter))
    error_face = np.zeros(len(action_filter))
    error_hand = np.zeros(len(action_filter))
    error_face_aligned = np.zeros(len(action_filter))
    error_hand_aligned = np.zeros(len(action_filter))

    for i, action in enumerate(action_filter):
        print(i, ' ', action)
        poses_valid, poses_valid_2d, actions_valid, cam_valid = fetch(["S8"], dataset, keypoints, [action],
                                                                      1)

        if poses_valid is not None:
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid, cam_valid),
                                      batch_size=256, shuffle=False,
                                      num_workers=24, pin_memory=True)

            error_eval_p1[i], error_body[i], error_face[i], error_hand[i], error_face_aligned[i], error_hand_aligned[
                i] = \
                evaluate(valid_loader, model_pos, device, kps_left, kps_right)
    print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(error_eval_p1).item()))
    print('Protocol #1   (MPJPE) action-wise average-body: {:.2f} (mm)'.format(np.mean(error_body).item()))
    print('Protocol #1   (MPJPE) action-wise average-face: {:.2f} (mm)'.format(np.mean(error_face).item()))
    print('Protocol #1   (MPJPE) action-wise average-hand: {:.2f} (mm)'.format(np.mean(error_hand).item()))
    print('Protocol #1   (MPJPE) action-wise average-face-aligned: {:.2f} (mm)'.format(
        np.mean(error_face_aligned).item()))
    print('Protocol #1   (MPJPE) action-wise average-hand-aligned: {:.2f} (mm)'.format(
        np.mean(error_hand_aligned).item()))

    return error_eval_p1





def evaluate(data_loader, model_pos, device, kps_left, kps_right):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_body = AverageMeter()
    epoch_loss_3d_face = AverageMeter()
    epoch_loss_3d_hand = AverageMeter()
    epoch_loss_3d_face_aligned = AverageMeter()
    epoch_loss_3d_hand_aligned = AverageMeter()
    # epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        # outputs_3d = model_pos(inputs_2d).cpu()
        ##### apply test-time-augmentation (following Videopose3d)


        body_3d, face_3d, left_hand_3d, right_hand_3d = model_pos(inputs_2d)
        outputs_3d = torch.cat((body_3d, face_3d, left_hand_3d, right_hand_3d), dim=1)



        outputs_3d = outputs_3d - (outputs_3d[:, 11:12, :] + outputs_3d[:, 12:13, :]) / 2
        targets_3d = targets_3d.to(device)
        face_3d_aligned = face_3d - face_3d[:, 30:31, :]

        hand_3d_aligned = torch.cat((left_hand_3d - left_hand_3d[:, :1, :], right_hand_3d - right_hand_3d[:, :1, :]),
                                    dim=1)

        hand_3d_aligned_gt = torch.cat(
            (targets_3d[:, 91:112, :] - targets_3d[:, 91:92, :], targets_3d[:, 112:, :] - targets_3d[:, 112:113, :]),
            dim=1)

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000, num_poses)
        epoch_loss_3d_body.update(mpjpe(outputs_3d[:, :23, :], targets_3d[:, :23, :]).item() * 1000, num_poses)
        epoch_loss_3d_face.update(mpjpe(outputs_3d[:, 23:91, :], targets_3d[:, 23:91, :]).item() * 1000, num_poses)
        epoch_loss_3d_hand.update(mpjpe(outputs_3d[:, 91:, :], targets_3d[:, 91:, :]).item() * 1000, num_poses)
        epoch_loss_3d_face_aligned.update(
            mpjpe(face_3d_aligned, (targets_3d - targets_3d[:, 53:54, ])[:, 23:91, :]).item() * 1000, num_poses)

        epoch_loss_3d_hand_aligned.update(mpjpe(hand_3d_aligned, hand_3d_aligned_gt).item() * 1000, num_poses)

        # epoch_loss_3d_pos_procrustes.update(
        #     p_mpjpe(outputs_3d.to('cpu').numpy(), targets_3d.to('cpu').numpy()).item() * 1000, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} |  body: {e3: .4f}| face: {e4: .4f}  ' \
                     '| hand: {e5: .4f} | face_aligned: {e6: .4f}| hand_aligned: {e7: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg,
                    e3=epoch_loss_3d_body.avg, e4=epoch_loss_3d_face.avg, e5=epoch_loss_3d_hand.avg,
                    e6=epoch_loss_3d_face_aligned.avg, e7=epoch_loss_3d_hand_aligned.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_body.avg, epoch_loss_3d_face.avg, \
        epoch_loss_3d_hand.avg, epoch_loss_3d_face_aligned.avg, epoch_loss_3d_hand_aligned.avg


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    main(parse_args())
