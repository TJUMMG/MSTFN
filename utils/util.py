import cv2
import torch
import numpy as np
from math import log10
from matplotlib import pyplot as plt


def cal_psnr(fake_img, real_img):

    diff = real_img - fake_img
    MSE = pow(diff, 2).mean()
    PSNR = -10*log10(MSE)

    return PSNR


def depose(img_in, img_path):
    # img_in = img_in / 2 + 0.5
    img_in = img_in.data.cpu().numpy()
    img_in = np.transpose(img_in, (1, 2, 0))
    img_in = np.clip(img_in, 0, 1)
    img_in = np.uint16(img_in * 65535.0)

    cv2.imwrite(img_path, img_in)


def get_lbd_sequence(idx, data_dir, quan):
    idx = str(idx)
    frame1 = data_dir + 'frame1/' + idx + '.png'
    frame2 = data_dir + 'frame2/' + idx + '.png'
    frame3 = data_dir + 'frame3/' + idx + '.png'
    frame4 = data_dir + 'frame4/' + idx + '.png'
    frame5 = data_dir + 'frame5/' + idx + '.png'

    # frame1 = data_dir + 'frame2/' + idx + '.png'
    # frame2 = data_dir + 'frame3/' + idx + '.png'
    # frame3 = data_dir + 'frame4/' + idx + '.png'
    # frame4 = data_dir + 'frame5/' + idx + '.png'
    # frame5 = data_dir + 'frame6/' + idx + '.png'

    # frame1 = data_dir + 'frame3/' + idx + '.png'
    # frame2 = data_dir + 'frame4/' + idx + '.png'
    # frame3 = data_dir + 'frame5/' + idx + '.png'
    # frame4 = data_dir + 'frame6/' + idx + '.png'
    # frame5 = data_dir + 'frame7/' + idx + '.png'

    img_l1 = g_lbd_img(frame1, quan)
    img_l2 = g_lbd_img(frame2, quan)
    img_l3 = g_lbd_img(frame3, quan)
    img_l4 = g_lbd_img(frame4, quan)
    img_l5 = g_lbd_img(frame5, quan)
    img = np.concatenate((img_l1, img_l2, img_l3, img_l4, img_l5), axis=2)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    img = img.unsqueeze(0)
    img = img.cuda()

    return img


def g_lbd_img(img_path, q):
    img = cv2.imread(img_path, 3)
    img = img // (2**q) * (2**q)
    img = img.astype('float32') / 65535.0
    return img


def depose(img, save_path):
    img = img.data.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = np.uint16(img * 65535.0)

    cv2.imwrite(save_path, img)


def save_map(img, save_path):
    img = img.data.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255

    # img = np.clip(img,0,1)
    img = np.uint8(img * 255.0)

    cv2.imwrite(save_path, img)


def plt_map(img):
    # norm = matplotlib.colors.Normalize(vmin=0.1,vmax=0.12)
    img = img.data.cpu().numpy()
    pmin = np.min(img)
    pmax = np.max(img)
    print(pmin, pmax)

    img = np.transpose(img, (1, 2, 0))
    plt.axis('off')

    plt.imshow(img, cmap='jet')
    plt.colorbar(shrink=0.4)
    # plt.imshow(img, cmap=plt.cm.gray)

    plt.savefig("/media/HardDisk_segate/fzw/code/MLSTN/results/experiments/review1_experiment/feature_map_experiment/s1g1.png",
                bbox_inches='tight', dpi=200)


def chop(img_input, model, c_h, c_w):

    batch, c, img_h, img_w = img_input.shape
    output = torch.zeros((batch, 3, img_h, img_w))

    print(batch, c, img_h, img_w)
    sub_res = []
    row = img_h // c_h
    col = img_w // c_w

    row_residual = img_h - row*c_h
    col_residual = img_w - col*c_w

    padding = 1
    up = 0
    dn = 0
    lf = 0
    rt = 0
    for r in range(row):
        for c in range(col):

            if r == 0 and c == 0:
                up = 0
                dn = padding
                lf = 0
                rt = padding

            elif r == 0 and c != 0 and c != (col-1):
                up = 0
                dn = padding
                lf = -padding
                rt = padding

            elif c == 0 and r != 0 and r != (row-1):
                up = -padding
                dn = padding
                lf = 0
                rt = padding

            elif r != 0 and c != 0 and r != (row-1) and c != (col-1):
                up = -padding
                dn = padding
                lf = -padding
                rt = padding

            elif r == (row-1) and c == 0:
                if row_residual == 0:
                    up = -padding
                    dn = 0
                    lf = 0
                    rt = padding
                else:
                    up = -padding
                    dn = padding
                    lf = 0
                    rt = padding

            elif c == (col-1) and r == 0:
                if col_residual == 0:
                    up = 0
                    dn = padding
                    lf = -padding
                    rt = 0
                else:
                    up = 0
                    dn = padding
                    lf = -padding
                    rt = padding

            elif r == (row-1) and c == (col-1):
                up = -padding
                lf = -padding
                if col_residual == 0:
                    rt = 0
                else:
                    rt = padding
                if row_residual == 0:
                    dn = 0
                else:
                    dn = padding

            elif r == (row-1) and c != 0 and c != (col-1):
                up = -padding
                lf = -padding
                rt = padding
                if row_residual == 0:
                    dn = 0
                else:
                    dn = padding
            elif c == (col-1) and r != 0 and r != (row-1):
                up = -padding
                dn = padding
                lf = -padding
                if col_residual == 0:
                    rt = 0
                else:
                    rt = padding
            img_in = img_input[:, :, r*c_h +
                               up: (r+1)*c_h + dn, c*c_w + lf: (c+1)*c_w + rt]
            in_bacth, in_ch, in_height, in_width = img_in.shape
            a = model(img_in)[:, :, -up: in_height - dn, -lf: in_width - rt]
            output[:, :, r*c_h: (r+1)*c_h, c*c_w: (c+1)*c_w] = a
            # print(output.shape)

            # sub_res.append(img_out)
        # res.append(sub_res)
    if row_residual != 0:
        for c in range(col):
            if c == 0:
                up = -padding
                dn = 0
                lf = 0
                rt = padding
            elif c == (col-1):
                up = -padding
                dn = 0
                lf = -padding
                if col_residual == 0:
                    rt = 0
                else:
                    rt = padding
            else:
                up = -padding
                dn = 0
                lf = -padding
                rt = padding
            img_in = img_input[:, :, row*c_h +
                               up: img_h, c*c_w + lf: (c+1)*c_w + rt]
            in_bacth, in_ch, in_height, in_width = img_in.shape
            output[:, :, row*c_h: img_h, c*c_w: (c+1)*c_w] = model(
                img_in)[:, :, -up: in_height - dn, -lf: in_width - rt]

    if col_residual != 0:
        for r in range(row):
            if r == 0:
                up = 0
                dn = padding
                lf = -padding
                rt = 0
            elif r == (row-1):
                up = -padding
                lf = -padding
                rt = 0
                if row_residual == 0:
                    dn = 0
                else:
                    dn = padding
            else:
                up = -padding
                dn = padding
                lf = -padding
                rt = 0
            img_in = img_input[:, :, r*c_h +
                               up: (r+1)*c_h + dn, col*c_w + lf: img_w]
            in_bacth, in_ch, in_height, in_width = img_in.shape
            output[:, :, r*c_h: (r+1)*c_h, col*c_w: img_w] = model(img_in)[:,
                                                                           :, -up: in_height - dn, -lf: in_width - rt]

    if row_residual != 0 and col_residual != 0:
        up = -padding
        dn = 0
        lf = -padding
        rt = 0
    img_in = img_input[:, :, row*c_h + up: img_h, col*c_w + lf: img_w]
    in_bacth, in_ch, in_height, in_width = img_in.shape
    output[:, :, row*c_h: img_h, col *
           c_w: img_w] = model(img_in)[:, :, -up: in_height, -lf: in_width]

    # for r in range(row):
    #     for c in range(col):
    #         output[:, :, r*c_h : (r+1)*c_h, c*c_w : (c+1)*c_w] = sub_res[r*col + c]

    return output
