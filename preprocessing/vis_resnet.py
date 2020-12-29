# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 6:41 下午
# @Author  : Yijia Zheng
# @FileName: vis_resnet.py

import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

savepath='../data/view_features'
if not os.path.exists(savepath):
    os.mkdir(savepath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet101(pretrained=True)
        self.model = model_ft

    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)
            draw_features(8, 8, x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))

            x = self.model.bn1(x)
            draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))

            x = self.model.relu(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))

            x = self.model.maxpool(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))

            x = self.model.layer1(x)
            draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))

            x = self.model.layer2(x)
            draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))

            x = self.model.layer3(x)
            draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))

            x = self.model.layer4(x)
            draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))

            x = self.model.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
            plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
            plt.savefig("{}/f10_fc.png".format(savepath))
            plt.clf()
            plt.close()

        return x


model=ft_net().cpu()

model.eval()
img = cv2.imread('../data/image/seg_train/buildings/4.jpg')
img = cv2.resize(img,(150,150))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
img = transform(img).cpu()
img = img.unsqueeze(0)
with torch.no_grad():
    start=time.time()
    out=model(img)
    print("total time:{}".format(time.time()-start))
    result=out.cpu().numpy()
    # ind=np.argmax(out.cpu().numpy())
    ind=np.argsort(result,axis=1)
    for i in range(5):
        print("predict:top {} = cls {} : score {}".format(i+1,ind[0,1000-i-1],result[0,1000-i-1]))
    print("done")

