clc;
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

re_path = '';
raw_path = '';
for idx = 1:50
    idx
%     raw_path_i = [raw_path num2str(idx) '.png'];
%     re_path_i = [re_path num2str(idx) '_ipad_16_8.png'];
    raw_path_i = [raw_path num2str(idx) 'HBD.png'];
    re_path_i = [re_path num2str(idx) '.png'];
    im=imread(raw_path_i);
    im=double(im);
    im_re=imread(re_path_i);
    im_re=double(im_re);
    % dv=(im-im_re)/(65535);
    dv=(im-im_re);
    MSE_pic(idx)=mean(mean(mean(dv.^2)));
    std_pic(idx)=std2(dv);
    PSNR_pic(idx)=10*log10(65535*65535/MSE_pic(idx));
    %PSNR2_pic(idx)=csnr_bits(16,im,im_re,0,0);
    SSIM_pic(idx)=cal_ssim_bits(16,im,im_re,0,0);
end

    
    
    PSNR=mean(PSNR_pic)
    %PSNR2=mean(PSNR2_pic)
    %std_mean=mean(std_pic)
    std_PSNR=std(PSNR_pic)
    SSIM=mean(SSIM_pic)
    std_SSIM=std(SSIM_pic)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     save('/home/d330/lpp/study/VSR_lpp/becnn/iftc_color/backup_dspres416_nored_adam_20/SSIM_PSNR2/SSIM_pic.mat','SSIM_pic');
%     save('/home/d330/lpp/study/VSR_lpp/becnn/iftc_color/backup_dspres416_nored_adam_20/SSIM_PSNR2/SSIM_mean.mat','SSIM');
%     save('/home/d330/lpp/study/VSR_lpp/becnn/iftc_color/backup_dspres416_nored_adam_20/SSIM_PSNR2/PSNR_pic.mat','PSNR_pic');
%     save('/home/d330/lpp/study/VSR_lpp/becnn/iftc_color/backup_dspres416_nored_adam_20/SSIM_PSNR2/PSNR_mean.mat','PSNR');
