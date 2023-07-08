# Multi-stage Spatio-Temporal Fusion Network for Fast and Accurate Video Bit-depth Enhancement

This is the official implementation of Multi-stage Spatio-Temporal Fusion Network for Fast and Accurate Video Bit-depth Enhancement (IEEE Transactions on Multimedia 2023) in Pytorch. 

Copyright(c) 2023 Jing Liu

```
If you use this code, please cite the following publication:
J. Liu, Z. Fan, Z. Yang, Y. Su, and X. Yang, "Multi-stage Spatio-Temporal Fusion Network for Fast and Accurate Video Bit-depth Enhancement", to appear in IEEE Transactions on Multimedia.
```


# Environment
You can run ```conda env create -f mstfn.yml``` to create the environment
- python=3.6
- ubuntu 20.04
- pytorch=1.6.0
- torchvision=0.7
- opencv-python==4.6

# How to Test
We provide folders "./dataset/test_frames_16bit" and "./dataset/test_frames_12bit" to realize 4-bit to 16-bit and 8-bit to 12-bit BDE tasks respectively. When testing, prepare the testing dataset, and modify the dataset path and other related content in the code. We provide a test sequence(16bit and 12bit) for testing. You can directly test on the testing frames by running
```
$ python test.py
```

# Note
- We provide recovery results in the folder "result". When testing, the predicted results are saved in the folder "test" .
- The files "./metrics/SSIM_PSNR.m" is used to calculate PSNR and SSIM.

