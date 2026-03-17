import os
import logging
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_trainloader, get_testloader
from dataloader import test_dataset2
import numpy as np
from model import USODModel
from utils import  History


image_root = ''
depth_root = ''
pse_root='t'
save_path = ''
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 超参数
batchsize =8
trainsize = 256
epoch =400
lr = 0.0001

model = USODModel()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, lr)
step = 0
best_mae = 1
best_epoch = 0


def structure_loss(pred, mask):
    # BCE loss
    k = nn.Softmax2d()
    weit = torch.abs(pred - mask)
    weit = k(weit)
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # IOU loss
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou)

def trace_loss(confidence, idx, history):
    loss1 = confidence
    loss2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)
    target, margin = history.get_target_margin(idx, idx2)
    target_nonzero = target.clone()
    loss1=torch.sum(loss1,dim=[2,3])
    loss2=torch.sum(loss2,dim=[2,3])
    loss2 = loss2+ (margin / target_nonzero)
    loss=torch.mean(torch.tanh((loss1-loss2))*(-margin.unsqueeze(1)))
    return loss


def train(train_loader, model, optimizer, epo, save_path,total_step,depth_history,rgb_history):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, pse_label, depth,name,idx) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            pse_label=pse_label.cuda()
            depth = depth.cuda()
            pred,rgb_out,depth_out,rgb_conf,depth_conf = model.forward_step(images,depth)
            #loss
            rgb_loss=structure_loss(rgb_out, pse_label)
            depth_loss=structure_loss(depth_out, pse_label)
            rgb_depth_loss = structure_loss(pred, pse_label)
            sal_loss=rgb_loss+depth_loss+rgb_depth_loss

            rgb_loss_detach=structure_loss(rgb_out, pse_label).detach()
            depth_loss_detach=structure_loss(depth_out, pse_label).detach()

            rgb_rank_loss=trace_loss(rgb_conf,idx,rgb_history)
            depth_rank_loss=trace_loss(depth_conf,idx,depth_history)

            depth_history.correctness_update(idx, depth_loss_detach, depth_conf.squeeze())
            rgb_history.correctness_update(idx, rgb_loss_detach, rgb_conf.squeeze())

            loss = sal_loss + 0.1* (depth_rank_loss + rgb_rank_loss)
            loss.backward()
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 50== 0 or i == total_step or i == 1:
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(epo, epoch, i, total_step, loss.data))
        if (epo) % 100== 0:
            torch.save(model.state_dict(), save_path + 'USOD_epoch_{}.pth'.format(epo))
        if epo == (epoch - 1):
            torch.save(model.state_dict(), save_path + 'USOD_epoch_{}.pth'.format(epo))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'USOD_epoch_{}.pth'.format(epo + 1))
        print('save checkpoints successfully!')
        raise

if __name__ == '__main__':
    # load data
    print('load data...')
    train_loader = get_trainloader(image_root, pse_root, depth_root, batchsize=batchsize,
                                   trainsize=trainsize)
    total_step = len(train_loader)
    depth_history = History(len(train_loader.dataset))
    rgb_history = History(len(train_loader.dataset))
    for epo in range(1, epoch):
        train(train_loader, model, optimizer, epo, save_path,total_step,depth_history,rgb_history)
        dataset = ''
        # if epo==150 or 200 or 250...
        if epo==150:
            hard_label1_root=''
            test_loader = test_dataset2(image_root, hard_label1_root,depth_root, trainsize, dataset)
            model.eval()
            with torch.no_grad():
                for i in range(test_loader.size):
                    image, depth, name, image_for_post = test_loader.load_data(dataset)
                    image = image.cuda()
                    depth = depth.cuda()
                    res,_,_,_,_ = model.forward_step(image,depth)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    cv2.imwrite(os.path.join(pse_root,name),res)

       





