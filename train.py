import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from modules.model import DICM
from modules.unet import Unet
import torch
from modules.trainer_dis import Trainer
from modules.compressor import Compressor

import config
from torch.utils.data import DataLoader
from dataset.PairKitti import PairKitti
from dataset.PairCityscape import PairCityscape
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--device", type=torch.device,default=accelerator.device, help="cuda device number")
args = parser.parse_args()

model_name = (
    f"{config.loss_type}-{config.data_config['dataset_name']}"
    f"-t{config.iteration_step}-b{config.beta}-vbr{config.vbr}"
    f"-{config.pred_mode}-{config.var_schedule}-aux{config.alpha}{config.aux_loss_type if config.alpha>0 else ''}{config.additional_note}"
)


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)

def load(model, idx=0, load_step=True):
        data = torch.load(
            f"{config.result_root}/basemodel/base.pt",
            map_location=torch.device(model.device),
        )

        if load_step:
            step = data["step"]
        model_to_load = accelerator.unwrap_model(model)

        try:
            model_to_load.module.load_state_dict(data["model"], strict=False)
        except :
            model_to_load.load_state_dict(data["model"], strict=False)
            
def data_load(config):
    path = config.data_config['data_path']
    resize = tuple([128,256])
    if config.data_config['dataset_name'] == 'KITTI':
        train_dataset = PairKitti(path=path, set_type='train', resize=resize)
        val_dataset = PairKitti(path=path, set_type='val', resize=resize)
        # test_dataset = PairKitti(path=path, set_type='test', resize=resize)
    elif config.data_config['dataset_name'] == 'Cityscape':
        train_dataset = PairCityscape(path=path, set_type='train', resize=resize)
        val_dataset = PairCityscape(path=path, set_type='val', resize=resize)
        # test_dataset = PairCityscape(path=path, set_type='test', resize=resize)
    else:
        raise Exception("Dataset not found")

    batch_size = config.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.n_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.n_workers)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=3)
    return train_loader,val_loader

def model_components(config):
    denoise = Unet(
        dim=config.embed_dim,
        channels=config.data_config["img_channel"],
        context_channels=config.context_channels,
        dim_mults=config.dim_mults,
        context_dim_mults=config.context_dim_mults
    )
    context = Compressor(
        dim=config.embed_dim,
        dim_mults=config.context_dim_mults,
        hyper_dims_mults=config.hyper_dim_mults,
        channels=config.data_config["img_channel"],
        out_channels=config.context_channels,
        vbr=config.vbr
    )

    context_cor = Compressor(
        dim=config.embed_dim,
        dim_mults=config.context_dim_mults,
        hyper_dims_mults=config.hyper_dim_mults,
        channels=config.data_config["img_channel"],
        out_channels=config.context_channels,
        vbr=config.vbr
    )
    context_w = Compressor(
        dim=config.embed_dim,
        dim_mults=config.context_dim_mults,
        hyper_dims_mults=config.hyper_dim_mults,
        channels=config.data_config["img_channel"],
        out_channels=config.context_channels,
        vbr=config.vbr
    )
    return denoise,context,context_cor,context_w

def main():

    train_loader,val_loader = data_load(config)

    denoise,context,context_cor,context_w = model_components(config)

    model = DICM(
        device=args.device,
        denoise_fn=denoise,
        context_fn=context,
        context_fn_cor = context_cor,
        context_w = context_w,
        clip_noise=config.clip_noise,
        num_timesteps=config.iteration_step,
        loss_type=config.loss_type,
        vbr=config.vbr,
        lagrangian=config.beta,
        pred_mode=config.pred_mode,
        aux_loss_weight=config.alpha,
        aux_loss_type=config.aux_loss_type,
        var_schedule=config.var_schedule
    ).to(args.device)
    
    if config.load_model:
        load(model,load_step=config.load_step)
        
    trainer = Trainer(
        rank=args.device,
        accelerator=accelerator,
        sample_steps=config.sample_steps,
        diffusion_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{model_name}/"),
        model_name=model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        sample_mode=config.sample_mode,
        lagrangian=config.beta,
    )

    trainer.train()


if __name__ == "__main__":
    main()
