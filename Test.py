import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from models.utlis import fps_subsample
from torch.utils.data import DataLoader
from utlis.dataloader import *
import torch
import torch.nn as nn
from utlis import meter
from tqdm import tqdm
from utlis.loss_utils import chamfer
from models.Model import *

DEVICE = 'cuda:0'

model = CSDN()
# model = nn.DataParallel(model)
model.to(device=DEVICE)
ckpt_path = f''
model.load_state_dict(torch.load(ckpt_path)['model'])

for i, cat in enumerate(['all']):
    losses_eval_cd = meter.AverageValueMeter()
    TEST_DATA = ViPCDataLoader('test_list.txt',
                               data_path='/path/ShapeNetViPC-Dataset',
                                   status='test', view_align=False, category=cat)
    test_loader = DataLoader(TEST_DATA,
                             batch_size=2,
                             num_workers=4,
                             shuffle=False,
                             drop_last=True)
    prefetcher = data_prefetcher(test_loader)
    views, pcs, pc_parts = prefetcher.next()
    iteration = 0
    pbar = tqdm(total=len(test_loader))
    model.eval()
    while views is not None:
        with torch.no_grad():
            iteration += 1
            views = views.to(device=DEVICE)
            pc_parts = pc_parts.to(device=DEVICE)
            pc_parts = fps_subsample(pc_parts, 2048)
            pcs = pcs.to(device=DEVICE)
            batch_size = views.size(0)
            fine,pcg,coarse = model(views,pc_parts)
            gt = fps_subsample(pcs, 2048)

            cd = chamfer(fine,gt)

            losses_eval_cd.add(cd.mean().item() * 1000)

            views, pcs, pc_parts= prefetcher.next()
            pbar.set_postfix(cd=(losses_eval_cd.value()[0]))
            pbar.update(1)
    pbar.close()
    print(cat, losses_eval_cd.value()[0])