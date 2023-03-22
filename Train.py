# coding=utf-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-C', '--cat', default='all')
parser.add_argument('-L', '--lr', default=5e-5)
parser.add_argument('-G', '--gpu', default='0')
parser.add_argument('-B', '--bz', default='4')
parser.add_argument('-E', '--epoch', default='31')
parser.add_argument('-EV', '--eval_epoch', default='1')
parser.add_argument('-N', '--num_workers', default='6')
parser.add_argument('-R', '--resume', default=False)

args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from models.Model import *
from models.utlis import fps_subsample
from torch.utils.data import DataLoader
from utlis.dataloader import *
import torch
import torch.nn as nn
from utlis import meter
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utlis.loss_utils import chamfer

if args.cat != None:
    CLASS = args.cat
else:
    CLASS = 'plane'

MODEL = f'CSDN'
FLAG = 'train'
DEVICE = 'cuda:0'
VERSION = '1.0'
LR = float(args.lr)
BATCH_SIZE = int(args.bz)
MAX_EPOCH = int(args.epoch)
NUM_WORKERS = int(args.num_workers)
EVAL_EPOCH = int(args.eval_epoch)
RESUME = args.resume

TIME_FLAG = time.time()
CKPT_RECORD_FOLDER = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{LR}_{CLASS}_{FLAG}_{TIME_FLAG}/record'
CKPT_FILE = f''
CONFIG_FILE = f'./log/{MODEL}/{MODEL}_{VERSION}_{BATCH_SIZE}_{LR}_{CLASS}_{FLAG}_{TIME_FLAG}/CONFIG.txt'

losses_all = meter.AverageValueMeter()
losses_cd = meter.AverageValueMeter()
losses_cd_down = meter.AverageValueMeter()
losses_cd_level0 = meter.AverageValueMeter()
losses_allrid = meter.AverageValueMeter()


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_view, self.next_pcs, self.next_pc_parts = next(self.loader)
        except StopIteration:
            self.next_view = None
            self.next_pcs = None
            self.next_pc_parts = None
            return
        with torch.cuda.stream(self.stream):
            self.next_view = self.next_view.cuda(non_blocking=True)
            self.next_pcs = self.next_pcs.cuda(non_blocking=True)
            self.next_pc_parts = self.next_pc_parts.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        view = self.next_view
        pcs = self.next_pcs
        pc_parts = self.next_pc_parts

        self.preload()
        return view, pcs, pc_parts


def save_record(epoch, prec1, net: nn.Module, optimizer_all):
    ckpt = dict(
        epoch=epoch,
        model=net.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
    )
    torch.save(ckpt, os.path.join(CKPT_RECORD_FOLDER, f'epoch{epoch}_{prec1:.4f}.pth'))



def train(epoch, model, g_optimizer, train_loader, board_writer):
    model.train()
    loss = None
    prefetcher = data_prefetcher(train_loader)
    views, pcs, pc_parts = prefetcher.next()
    iteration = 0
    pbar = tqdm(total=len(train_loader))
    while views is not None:
        iteration += 1
        if iteration + epoch * len(train_loader) < 1000:
            alpha = 0.01
        elif iteration + epoch * len(train_loader) < 3000:
            alpha = 0.1
        elif iteration + epoch * len(train_loader) < 5000:
            alpha = 1.0
        else:
            alpha = 2.0
        views = views.to(device=DEVICE)
        pc_parts = pc_parts.to(device=DEVICE)
        pc_parts = fps_subsample(pc_parts,2048)
        pcs = pcs.to(device=DEVICE)
        batch_size = views.size(0)
        g_optimizer.zero_grad()
        gt_2048 = fps_subsample(pcs, 2048)
        coarse,pcg,fine = model(views,pc_parts)
        # loss_total, losses, gts = get_loss_snow(pcds_pred,
        #                                    pc_parts,
        #                                    gt_2048,
        #                                    sqrt=True)
        cd_fine = chamfer(fine, gt_2048)
        cd_coarse = chamfer(pcg,gt_2048)
        loss_total = alpha*cd_fine+cd_coarse
        loss_total.backward()
        g_optimizer.step()

        board_writer.add_scalar('loss/loss_all', loss_total,
                                global_step=iteration + epoch * len(train_loader))
        board_writer.add_scalar('loss/loss_fine', cd_fine,
                                global_step=iteration + epoch * len(train_loader))
        board_writer.add_scalar('loss/loss_coarse', cd_coarse,
                                global_step=iteration + epoch * len(train_loader))
        board_writer.add_scalar('lr', g_optimizer.state_dict()['param_groups'][0]['lr'],
                                global_step=iteration + epoch * len(train_loader))

        views, pcs, pc_parts = prefetcher.next()
        pbar.set_description('[Epoch %d/%d]' % (epoch, int(args.epoch)))
        pbar.set_postfix(loss=float(loss_total),cd_fine=float(cd_fine*1000))
        pbar.update(1)

    if epoch % 1 == 0:
        save_record(epoch, losses_all.value()[0], model, g_optimizer)
    pbar.close()
    return loss


def model_eval(epoch, criterion, model, test_loader, board_writer):
    losses_eval_cd = meter.AverageValueMeter()
    prefetcher = data_prefetcher(test_loader)
    views, pcs, pc_parts = prefetcher.next()
    # pc_parts_t = pc_parts.permute(0, 2, 1)
    iteration = 0
    pbar = tqdm(total=len(test_loader))
    model.eval()
    while views is not None:
        with torch.no_grad():
            iteration += 1
            views = views.to(device=DEVICE)
            pc_parts = pc_parts.to(device=DEVICE)
            pcs = pcs.to(device=DEVICE)
            batch_size = views.size(0)
            pc_parts = fps_subsample(pc_parts, 2048)
            pcs = pcs.to(device=DEVICE)
            gt_2048 = fps_subsample(pcs, 2048)
            coarse,pcg,fine = model(views,pc_parts)

            loss_cd = criterion(fine,gt_2048)

            losses_eval_cd.add(loss_cd * 1000)

            views, pcs, pc_parts = prefetcher.next()
            pbar.set_postfix(loss=(losses_eval_cd.value()[0]))
            pbar.update(1)
    board_writer.add_scalar('test/loss', losses_eval_cd.value()[0],
                            global_step=epoch)
    pbar.close()
    return losses_eval_cd.value()[0]


def main():
    print('--------------------')
    print('Training')
    print(f'Tring Class: {CLASS}')
    print('--------------------')
    model = CSDN()

    model.to(device=DEVICE)
    # G = nn.DataParallel(G)

    g_optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.1, last_epoch=-1)

    TRAIN_DATA = ViPCDataLoader('train_list.txt',
                                data_path='/path/ShapeNetViPC-Dataset', status='train',
                                view_align=False, category=args.cat)
    TEST_DATA = ViPCDataLoader('test_list.txt', data_path='/path/ShapeNetViPC-Dataset',
                               status='test', view_align=False, category=args.cat)

    train_loader = DataLoader(TRAIN_DATA,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(TEST_DATA,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=True,
                             drop_last=True)

    resume_epoch = 0
    board_writer = SummaryWriter(comment=f'{MODEL}_{VERSION}_{BATCH_SIZE}_{LR}_{FLAG}_{CLASS}_{TIME_FLAG}')

    if RESUME:
        print("resuming...")
        ckpt_path = CKPT_FILE
        ckpt_dict = torch.load(ckpt_path)
        model.load_state_dict(ckpt_dict['model'])
        g_optimizer.load_state_dict(ckpt_dict['optimizer_all'])
        g_optimizer.param_groups[0]['lr']= 5e-6
        resume_epoch = ckpt_dict['epoch']

    if not os.path.exists(os.path.join(CKPT_RECORD_FOLDER)):
        os.makedirs(os.path.join(CKPT_RECORD_FOLDER))

    with open(CONFIG_FILE, 'w') as f:
        f.write('RESUME:' + str(RESUME) + '\n')
        f.write('FLAG:' + str(FLAG) + '\n')
        f.write('DEVICE:' + str(DEVICE) + '\n')
        f.write('LR:' + str(LR) + '\n')
        f.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
        f.write('MAX_EPOCH:' + str(MAX_EPOCH) + '\n')
        f.write('NUM_WORKERS:' + str(NUM_WORKERS) + '\n')
        f.write('CLASS:' + str(CLASS) + '\n')

    best_loss = 9999
    best_epoch = 0

    for epoch in range(resume_epoch, resume_epoch + MAX_EPOCH):
        losses = train(epoch, model, g_optimizer, train_loader, board_writer)
        scheduler.step()
        if epoch % EVAL_EPOCH == 0:
            loss = model_eval(epoch, chamfer, model, test_loader, board_writer)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
            print('This Epoch:', loss)
        print('****************************')
        print('Best Performance: Epoch', best_epoch, ' CD_LOSS:', best_loss)
        print('****************************')

    print('Train Finished!')


if __name__ == '__main__':
    main()