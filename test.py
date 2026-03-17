import torch
import os
import cv2
from model import USODModel
from dataloader import test_dataset2



testsize=256
model = USODModel()
model.load_state_dict(torch.load(''))
model.cuda()
model.eval()
test_datasets=['']

for dataset in test_datasets:
    save_path =''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root=''
    depth_root = ''
    test_loader = test_dataset2(image_root,  image_root,depth_root, testsize, dataset)

    for i in range(test_loader.size):
        image,  depth, name, image_for_post = test_loader.load_data(dataset)
        image = image.cuda()
        depth = depth.cuda()
        res,_,_,_,_= model.forward_step(image,depth)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(os.path.join(save_path,name),res*255)
        print("write pse_label :", name)

    print('Test Done!')