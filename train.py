import time
import os

# os.environ['CUDA_VISIBLE_DEVICES']='1'

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from nets.unet import Unet
from nets.unet_training import CE_Loss, Dice_loss
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.metrics import f_score


# print(torch.cuda.current_device())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name())
# print(torch.cuda.is_available())

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    net = net.train()
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(imgs)
            loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                's/step': waste_time,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs = net(imgs)
                val_loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    val_loss = val_loss + main_dice
                # -------------------------------#
                #   f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score': val_total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))

    f = open('D:/1chengxu/unet-pytorch-main-old/logs4/loss.txt', 'a')
    f.write(str(total_loss / (epoch_size + 1)) + ',')
    f.close()

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs4/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))


if __name__ == "__main__":
    start_time = time.time()

    log_dir = "logs4/"
    # ------------------------------#

    # ------------------------------#
    inputs_size = [512, 512, 3]
    # inputs_size1 = [512,512,3]
    # ---------------------#


    # ---------------------#
    NUM_CLASSES = 2
    # --------------------------------------------------------------------#

    # ---------------------------------------------------------------------#
    dice_loss = True
    # -------------------------------#

    # -------------------------------#
    pretrained = True
    # -------------------------------#

    # -------------------------------#
    Cuda = True

    model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train()

    # -------------------------------------------#

    # -------------------------------------------#
    model_path = r"model_data/unet_voc.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    if Cuda:
        device_ids = [0]
        net = torch.nn.DataParallel(model, device_ids=device_ids)
        cudnn.benchmark = True
        net = net.cuda()
        print(torch.cuda.current_device())  # 输出gpu名字
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name())
        print(torch.cuda.is_available())


    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt", "r") as f:
        train_lines = f.readlines()


    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt", "r") as f:
        val_lines = f.readlines()

    # ------------------------------------------------------#

    # ------------------------------------------------------#
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Interval_Epoch = 40
        Batch_size = 1

        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        for param in model.vgg.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Interval_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-5
        Interval_Epoch = 41
        Epoch = 80
        Batch_size = 1

        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        for param in model.vgg.parameters():
            param.requires_grad = True

        for epoch in range(Interval_Epoch, Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, Cuda)
            lr_scheduler.step()

    # if True:
    #     lr = 1e-6
    #     Interval_Epoch = 81
    #     Epoch = 100
    #     Batch_size = 1
    #
    #     optimizer = optim.Adam(model.parameters(), lr)
    #     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    #
    #     train_dataset = DeeplabDataset(train_lines, inputs_size, NUM_CLASSES, True)
    #     val_dataset = DeeplabDataset(val_lines, inputs_size, NUM_CLASSES, False)
    #     gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
    #                      drop_last=True, collate_fn=deeplab_dataset_collate)
    #     gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
    #                          drop_last=True, collate_fn=deeplab_dataset_collate)
    #
    #     epoch_size = max(1, len(train_lines) // Batch_size)
    #     epoch_size_val = max(1, len(val_lines) // Batch_size)
    #
    #     for param in model.vgg.parameters():
    #         param.requires_grad = True
    #
    #     for epoch in range(Interval_Epoch, Epoch):
    #         fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, Cuda)
    #         lr_scheduler.step()

    end_time = time.time()
    print('------------------------------')
    print('time：%s' % (end_time - start_time))

