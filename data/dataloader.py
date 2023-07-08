import os
import cv2
import math
import torch
import numpy as np
import torch.utils.data as data

class LiverDataset(data.Dataset):
    def __init__(self, root, quanti_bit,):
        self.quanti = math.pow(2.0, quanti_bit)
        self.x = []
        frames = os.listdir(root + 'frame3/')
        num = len(frames)
        for i in range(num):
            root1 = root + 'frame1/' + str(i+1) + '.png'
            root2 = root + 'frame2/' + str(i+1) + '.png'
            root3 = root + 'frame3/' + str(i+1) + '.png'
            root4 = root + 'frame4/' + str(i+1) + '.png'
            root5 = root + 'frame5/' + str(i+1) + '.png'

            self.x.append([root1, root2, root3, root4, root5])


    def __getitem__(self, index):
        # img_x,input
        img1 = cv2.imread(self.x[index][0],3)
        img1 = img1.astype('float32')
        img_x1 = np.floor(img1/self.quanti)*self.quanti
        img_x1 = img_x1/65535.0

        img2 = cv2.imread(self.x[index][1],3)
        img2 = img2.astype('float32')
        img_x2 = np.floor(img2/self.quanti)*self.quanti
        img_x2 = img_x2/65535.0

        img3 = cv2.imread(self.x[index][2],3)
        img3 = img3.astype('float32')
        img_x3 = np.floor(img3/self.quanti)*self.quanti
        img_x3 = img_x3/65535.0

        img4 = cv2.imread(self.x[index][3],3)
        img4 = img4.astype('float32')
        img_x4 = np.floor(img4/self.quanti)*self.quanti
        img_x4 = img_x4/65535.0

        img5 = cv2.imread(self.x[index][4],3)
        img5 = img5.astype('float32')
        img_x5 = np.floor(img5/self.quanti)*self.quanti
        img_x5 = img_x5/65535.0

        img_y3 = img3/65535.0

        img_x1 = torch.from_numpy(img_x1.transpose((2, 0, 1)))
        img_x2 = torch.from_numpy(img_x2.transpose((2, 0, 1)))
        img_x3 = torch.from_numpy(img_x3.transpose((2, 0, 1)))
        img_x4 = torch.from_numpy(img_x4.transpose((2, 0, 1)))
        img_x5 = torch.from_numpy(img_x5.transpose((2, 0, 1)))
        # (C, H, W)---->(5C, H, W)
        img_x = np.concatenate((img_x1, img_x2, img_x3, img_x4, img_x5), axis=0)
        # (C, H, W)---->(5,C, H, W)
        # img_x = np.stack((img_x1, img_x2, img_x3, img_x4, img_x5), axis=0)
        img_y3 = torch.from_numpy(img_y3.transpose((2, 0, 1)))

        return img_x, img_y3

    def __len__(self):
        return len(self.x)

    # def _get_patch(self, img_x1, img_x2, img_x3, img_x4, img_x5, img_y, patch_size_x, patch_size_y):
    #     ih, iw = img_x1.shape[:2]
    #
    #     ix = random.randrange(0, iw - patch_size_x + 1)
    #     iy = random.randrange(0, ih - patch_size_y + 1)
    #
    #     img_x1 = img_x1[iy:iy + patch_size_y, ix:ix + patch_size_x, :]
    #     img_x2 = img_x2[iy:iy + patch_size_y, ix:ix + patch_size_x, :]
    #     img_x3 = img_x3[iy:iy + patch_size_y, ix:ix + patch_size_x, :]
    #     img_x4 = img_x4[iy:iy + patch_size_y, ix:ix + patch_size_x, :]
    #     img_x5 = img_x5[iy:iy + patch_size_y, ix:ix + patch_size_x, :]
    #
    #     img_y = img_y[iy:iy + patch_size_y, ix:ix + patch_size_x, :]
    #
    #     return img_x1, img_x2, img_x3, img_x4, img_x5, img_y