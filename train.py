import os

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'

import torch
from torch import nn
 
    
if __name__ == '__main__':
    num_classes = 3  
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1
    while epoch < 50:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            #_image = image[0]
            #_segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            #_out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([image[0], segment_image[0], out_image[0]], dim=0)
            save_image(img, f'{save_path}/{i}.png')
            if i % 100 == 0:
                torch.save(net.state_dict(), weight_path)
                print('save successfully!')
        epoch += 1
