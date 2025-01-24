import torch
from pathlib import Path
from torch.optim import Adam, AdamW

import numpy as np
from .components import cycle
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ms_ssim
import time

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# trainer class
class Trainer(object):
    def __init__(
        self,
        accelerator,
        rank,
        sample_steps,
        diffusion_model,
        train_loader,
        val_loader,
        scheduler_function,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        sample_mode="ddpm",
        lagrangian = 1,
    ):
        super().__init__()
        self.model = diffusion_model
        self.sample_mode = sample_mode
        self.sample_steps = sample_steps
        self.save_and_sample_every = save_and_sample_every
        self.accelerator = accelerator
        self.train_num_steps = train_num_steps

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lagrangian = lagrangian
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)

        self.step = 0
        self.device = accelerator.device
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name
        self.model, self.opt,self.train_loader,self.val_loader,self.scheduler = self.accelerator.prepare(self.model, self.opt, self.train_loader,self.val_loader,self.scheduler)

    def save(self):
        self.accelerator.wait_for_everyone()
        model = self.accelerator.unwrap_model(self.model)
        data = {
            "step": self.step,
            "model": model.state_dict(),
            # "ema": self.ema_model.module.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) % 3
        self.accelerator.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))


    # def load(self, idx=0, load_step=True):
    #     data = torch.load(
    #         "/home/test/hq/base/city/big-l1-Cityscape-d64-t20000-b0.056-vbrFalse-noise-linear-aux0.9lpips/big-l1-Cityscape-d64-t20000-b0.056-vbrFalse-noise-linear-aux0.9lpips_0.pt",
    #         map_location=lambda storage, loc: storage,
    #     )

    #     if load_step:
    #         self.step = data["step"]
    #     try:
    #         self.model.module.load_state_dict(data["model"], strict=False)
    #     except:
    #         self.model.load_state_dict(data["model"], strict=False)
    #     # self.ema_model.module.load_state_dict(data["ema"], strict=False)

    def train(self):

        mse = torch.nn.MSELoss(reduction='mean')
        mse = mse.to(self.device)
        st=time.time()
        for epoch in range(self.train_num_steps):
            self.model.train()
            if (epoch >= self.scheduler_checkpoint_step) and (epoch != 0):
                self.scheduler.step()
            # if self.accelerator.is_main_process:
            #     print(epoch)
            for i, data in enumerate(iter(self.train_loader)):
                self.opt.zero_grad()
                img, cor_img, _, _ = data
                img = img.float().to(self.device)
                cor_img = cor_img.float().to(self.device)

                loss_x,loss_y,aloss_x,aloss_y = self.model(img * 2.0 - 1.0,cor_img * 2.0 - 1.0)
                # loss = loss_x + loss_y

                # loss.backward()
                # aloss= aloss_x + aloss_y

                # aloss.backward()
                self.accelerator.backward(loss_x+loss_y+aloss_x+aloss_y)
                if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()

            if (epoch % self.save_and_sample_every == 0):

                val_loss = []
                val_mse = []
                val_mse_y = []
                val_msssim = []
                val_bpp = []
                val_transmitted_bpp = []
                val_distortion = []

                self.model.eval()
                with torch.no_grad():

                    for i, data in enumerate(iter(self.val_loader)):
                        # img = input image, cor_img = side information/correlated image (designated y in the paper)
                        img, cor_img, _, _ = data
                        img = img.float().to(self.device)
                        cor_img = cor_img.float().to(self.device)
                        # img = img.unsqueeze(0)/255.0
                        # cor_img = cor_img.unsqueeze(0)/255.0

                        compressed_x,compressed_y, bpp,transmitted_bpp = self.model.module.compress(
                            img * 2.0 - 1.0,
                            cor_img * 2.0 - 1.0,
                            sample_steps=200,
                            sample_mode="ddim",
                            bpp_return_mean=False,
                            init=torch.randn_like(img) * 0.8
                        )

                        compressed_x = compressed_x.clamp(-1, 1) / 2.0 + 0.5
                        compressed_y = compressed_y.clamp(-1, 1) / 2.0 + 0.5

                        mse_dist = mse(img, compressed_x)
                        msssim = ms_ssim(img.clone(), compressed_x.clone(), data_range=1.0, size_average=True,win_size=7)
                        #msssim = ms_ssim(img.clone().cpu(), compressed_x.clone().cpu(), data_range=1.0, size_average=True,win_size=7)
                        msssim_db = msssim

                        mse_dist_y = mse(cor_img, compressed_y)

                        distortion = (1 - ms_ssim(img, compressed_x, data_range=1.0, size_average=True,win_size=7))
                        #distortion = (1 - ms_ssim(img.cpu(), compressed_x.cpu(), data_range=1.0, size_average=True,win_size=7))
                        distortion += (1 - ms_ssim(cor_img, compressed_y, data_range=1.0, size_average=True,win_size=7))
                        #distortion += (1 - ms_ssim(cor_img.cpu(), compressed_y.cpu(), data_range=1.0, size_average=True,win_size=7))

    
                        loss = self.lagrangian * distortion * (255 ** 2) + bpp  # multiplied by (255 ** 2) for distortion scaling

                        val_mse_y.append(torch.tensor(mse_dist_y.item()).to(self.accelerator.device))
                        val_transmitted_bpp.append(torch.tensor(torch.mean(transmitted_bpp).item()).to(self.accelerator.device))
                        val_mse.append(torch.tensor(mse_dist.item()).to(self.accelerator.device))
                        val_bpp.append(torch.tensor(torch.mean(bpp).item()).to(self.accelerator.device))
                        val_loss.append(torch.tensor(torch.mean(loss).item()).to(self.accelerator.device))
                        val_msssim.append(torch.tensor(torch.mean(msssim_db).item()).to(self.accelerator.device))
                        val_distortion.append(torch.tensor(torch.mean(distortion).item()).to(self.accelerator.device))

                gathered_val_loss = self.accelerator.gather_for_metrics(torch.stack(val_loss))
                gathered_val_mse = self.accelerator.gather_for_metrics(torch.stack(val_mse))
                gathered_val_mse_y = self.accelerator.gather_for_metrics(torch.stack(val_mse_y))
                gathered_val_transmitted_bpp = self.accelerator.gather_for_metrics(torch.stack(val_transmitted_bpp))
                gathered_val_bpp = self.accelerator.gather_for_metrics(torch.stack(val_bpp))
                gathered_val_msssim = self.accelerator.gather_for_metrics(torch.stack(val_msssim))
                gathered_val_distortion = self.accelerator.gather_for_metrics(torch.stack(val_distortion))


                val_loss_to_track = gathered_val_loss.mean().item()
                avg_bpp = gathered_val_bpp.mean().item()
                avg_t_bpp = gathered_val_transmitted_bpp.mean().item()
                avg_distortion = gathered_val_distortion.mean().item()
                avg_mse = gathered_val_mse.mean().item()
                avg_mse_y = gathered_val_mse_y.mean().item()
                avg_msssim = gathered_val_msssim.mean().item()
                
                tracking = ['Epoch {}:'.format(epoch + 1),
                            'Loss= {:.4f},'.format(val_loss_to_track),
                            'BPP= {:.4f},'.format(avg_bpp),
                            'Distortion= {:.4f},'.format(avg_distortion),
                            'Transmitted BPP = {:.4f},'.format(avg_t_bpp),
                            'PSNR= {:.4f},'.format(10 * np.log10(1 / avg_mse)),
                            'PSNR_y = {:.4f},'.format(10 * np.log10(1 / avg_mse_y)),
                            'MS-SSIM= {:.4f}'.format(avg_msssim)]
                end_time = time.time()
                execution_time = end_time - st
                formatted_time = format_duration(execution_time)
                
                if self.accelerator.is_main_process:
                    print(f"{formatted_time} "+str(self.opt.param_groups[0]['lr'])+"  "+" ".join(tracking))
                self.save()
            self.save()
        print("training completed")
