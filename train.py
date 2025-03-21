# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction,DetailFeatureExtraction2,FeatureExtraction
from utils.dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc,Fusionloss2
import kornia
from pytorch_msssim import ms_ssim


'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,4,5'
criteria_fusion = Fusionloss()
criteria_fusion2 = Fusionloss2()
model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 120 # total epoch
epoch_gap = 60  # epoches of Phase I 

loaded_phase1 = False

lr = 1e-3
weight_decay = 0
batch_size = 20
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 30
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
#DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction2()).to(device)
FuseLayer=nn.DataParallel(FeatureExtraction()).to(device)
# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(
    FuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


# data loader
trainloader = DataLoader(H5Dataset("/mnt/storage/wjh/shy/data_h5/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

if os.path.exists('phase1_checkpoint.pth') and loaded_phase1:
    checkpoint = torch.load('phase1_checkpoint.pth')
    DIDF_Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
    DIDF_Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
    optimizer1.load_state_dict(checkpoint['optimizer1'])
    optimizer2.load_state_dict(checkpoint['optimizer2'])
    scheduler1.load_state_dict(checkpoint['scheduler1'])
    scheduler2.load_state_dict(checkpoint['scheduler2'])
    print("Loaded saved model for Phase 2 training.")
else:
    print("No saved Phase 1 model found.")


for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR,data_VIS2,data_IR2) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        # BaseFuseLayer.train()
        # DetailFuseLayer.train()
        FuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        # BaseFuseLayer.zero_grad()
        # DetailFuseLayer.zero_grad()
        FuseLayer.zero_grad()
    
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()

        if epoch < epoch_gap: #Phase I
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS2)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR2)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS2, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR2, feature_I_B, feature_I_D)
            # data_IR_hat, _ = DIDF_Decoder(None, feature_I_B, feature_I_D)
            # data_VIS_hat, _ = DIDF_Decoder(None, feature_V_B, feature_V_D)
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))
            L1loss_V=10*L1Loss(data_VIS,data_VIS_hat)
            L1loss_I=10*L1Loss(data_IR,data_IR_hat)
            mse_loss_V2=5 * Loss_ssim(data_VIS, data_VIS_hat)+L1loss_V
            mse_loss_I2=5 * Loss_ssim(data_IR, data_IR_hat)+L1loss_I
            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            #loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   #mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss
            
            #loss = coeff_mse_loss_VF *mse_loss_V2 + coeff_mse_loss_IF*mse_loss_I2 + coeff_decomp * loss_decomp
            loss = 100*(MSELoss(data_VIS, data_VIS_hat) + MSELoss(data_IR, data_IR_hat)) + 100*coeff_decomp * loss_decomp
            print("mseloss:",100*(MSELoss(data_VIS, data_VIS_hat) + MSELoss(data_IR, data_IR_hat)))
            print("decomploss:",loss_decomp)
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            DIDF_Encoder.eval()
            DIDF_Decoder.eval()

            for param in DIDF_Encoder.parameters():
                param.requires_grad = False
            for param in DIDF_Decoder.parameters():
                param.requires_grad = False
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS2)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR2)
            #feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            #feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            feature_F_D1,feature_F_B1=FuseLayer(feature_I_B+feature_V_B)
            feature_F_D2,feature_F_B2=FuseLayer(feature_I_D+feature_V_D)
            feature_F_B=feature_F_B1+feature_F_B2
            feature_F_D=feature_F_D1+feature_F_D2
            data_Fuse, feature_F = DIDF_Decoder(data_VIS2, feature_F_B, feature_F_D)  
            # data_Fuse, feature_F = DIDF_Decoder(None, feature_F_B, feature_F_D)
            VIS_hat,_=DIDF_Decoder(data_VIS2, feature_V_B, feature_V_D)
            IR_hat,_=DIDF_Decoder(data_IR2,feature_I_B,feature_I_D)

            L1loss_V_hat=10*L1Loss(data_VIS,VIS_hat)
            L1loss_I_hat=10*L1Loss(data_IR,IR_hat)
            #mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            #mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            fusionloss, _,_  = criteria_fusion(VIS_hat, IR_hat, data_Fuse)
            #fusionloss=criteria_fusion2(data_VIS, data_IR, data_Fuse)
            #loss = fusionloss + coeff_decomp * loss_decomp + L1loss_V_hat+L1loss_I_hat
            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            # nn.utils.clip_grad_norm_(
            #     DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            # nn.utils.clip_grad_norm_(
            #     DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                FuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            # optimizer1.step()  
            # optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )
        with open('training_log.txt', 'a') as f:
            f.write(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s\n"
                % (
                    epoch ,
                    num_epochs,
                    i ,
                    len(loader['train']),
                    loss.item(),
                    time_left,
                )
            )
    if epoch == epoch_gap - 1:
        checkpoint = {
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'scheduler1': scheduler1.state_dict(),
            'scheduler2': scheduler2.state_dict(),
        }
        torch.save(checkpoint, 'phase1_checkpoint.pth')
        print(f"Saved Phase 1 model at epoch {epoch}")
    # adjust the learning rate
    if epoch<epoch_gap:
        scheduler1.step()  
        scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/CDDFuse_"+timestamp+'.pth'))


