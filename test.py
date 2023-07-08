import os
import cv2
import time
import utils
import torch
import argparse
import numpy as np
from model.mlstn import vbde_net

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--ckp', type=str, default="./checkpoint/4_16/netG_weights_epo61.pth")
    parse.add_argument('--data_dir', type=str, default="./dataset/test_frames_16bit")
    parse.add_argument('--result_path', type=str, default='./results/')
    parse.add_argument('--save_file', type=str, default='test/')
    parse.add_argument("--quanti_bit", type=int, default=12)
    args = parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args1 = vars(args)
    f_path = args.result_path + args.save_file
    if os.path.isdir(f_path) == False:
        os.makedirs(f_path)
    f = open(f_path + 'test_logging.txt', 'w')
    for k, v in args1.items():
        f.write(k + ':' + str(v))
        f.write('\n\n')
    f.close()

    model = vbde_net().cuda()
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(args.ckp).items()})
    model.eval()
    data_dir = args.data_dir
    result_path = args.result_path
    save_file = args.save_file

    if os.path.isdir(result_path) == False:
        os.mkdir(result_path)
    result_hbd1 = os.path.join(result_path, save_file)
    if os.path.isdir(result_hbd1) == False:
        os.mkdir(result_hbd1)

    total_time = 0
    num = 1

    for i in range(1, num+1):
        i = str(i)
        print('processing the', i, 'th image')
        img_input = utils.get_lbd_sequence(i, data_dir, args.quanti_bit)
        with torch.no_grad():
            t0 = time.time()
            outputs = utils.chop(img_input, model, 150, 150)
            # outputs = model(img_input)
            outputs = outputs.cuda()
            t1 = time.time()
            time_df = t1-t0
            total_time += time_df
            print("it takes {:.2f} seconds to get the advanced image".format(time_df))
            output = torch.squeeze(outputs)
            result = result_hbd1 + i + '.png'
            utils.depose(output, result)

    print("it takes average {:.2f} seconds to get the advanced image".format(total_time/num))
