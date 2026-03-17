import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root, trainsize):
        self.trainsize = trainsize
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.jpeg')
                    or f.endswith('.png') or f.endswith('.PNG')]
        self.images = []
        self.depths=[]
        for file in self.gts:
            filename=file.split('/')[-1].split('.')[0]
            self.images.append(os.path.join(image_root,filename+'.PNG'))
            self.depths.append(os.path.join(depth_root,filename+'.PNG'))
        self.gts = sorted(self.gts)
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if gt.size != image.size:
            gt = gt.resize(image.size)
        depth = self.rgb_loader(self.depths[index])
        depth = depth.resize(image.size)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        name = self.images[index].split('/')[-1]
        return image,gt, depth,name,index

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class test_dataset1(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.jpeg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]

        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.depths = sorted(self.depths)
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.gt_transform =transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if gt.size != image.size:
            gt = gt.resize(image.size)
        depth = self.rgb_loader(self.depths[index])
        depth = self.depths_transform(depth)
        image = self.transform(image)
        gt = self.gt_transform(gt)


        return image, gt,  depth

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



    def __len__(self):
        return self.size

class test_dataset2:
    def __init__(self, image_root,hard_label_root ,depth_root, testsize, set_name):
        self.dataset_name = set_name
        self.testsize = testsize
        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.PNG') or f.endswith('.png') or f.endswith('.bmp')]
        self.hard_sample = [hard_label_root + f for f in os.listdir(hard_label_root) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.PNG') or f.endswith('.png') or f.endswith('.bmp')]
        # print(self.hard_sample)
        self.images = []
        self.depth=[]
        for file in self.hard_sample:
            filename = file.split('/')[-1].split('.')[0]
            self.images.append(os.path.join(image_root, filename+'.PNG'))#训练PNG
            self.depth.append(os.path.join(depth_root, filename+'.PNG'))#训练PNG

        # self.depth = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.jpeg')
        #             or f.endswith('.png') or f.endswith('.PNG')]
        self.images = sorted(self.images)

        self.depth = sorted(self.depth)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.depth_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self, set_name):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        depth = self.rgb_loader(self.depth[self.index])
        depth= self.depth_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])

        self.index += 1
        self.index = self.index % self.size
        return image,  depth,  name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



    def __len__(self):
        return self.size


def get_trainloader(image_root, gt_root,depth_root,  batchsize, trainsize, shuffle=True, num_workers=1, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root,depth_root,  trainsize)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_testloader(image_root, gt_root,depth_root,   batchsize, testsize, shuffle=False, num_workers=1, pin_memory=True):
    dataset = test_dataset1(image_root, gt_root, depth_root,  testsize)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

