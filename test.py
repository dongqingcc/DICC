
import argparse
import os
import torch
import torchvision
import numpy as np
# import pathlib
import config
from modules.model import DICM
from modules.unet import Unet
from modules.compressor import Compressor
from pytorch_msssim import ms_ssim
import PIL.Image as Image
import pandas as pd
from dataset.PairKitti import PairKitti
from dataset.PairCityscape import PairCityscape
from torch.utils.data import DataLoader
import lpips
# import lpips
# from scipy import linalg
# from torchvision.models import inception_v3
# from torchvision import transforms
# from torch.autograd import Variable
# from model.PieAPPv0pt1_PT import PieAPP
# from pytorch_fid import fid_score
from DISTS_pytorch import DISTS



parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, required=True) # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=200) # number of denoising step
parser.add_argument("--device", type=int, default=0) # gpu device index
parser.add_argument("--img_dir", type=str, default='/data/code/data/Cityscape/')
parser.add_argument("--out_dir", type=str, default='./result')
parser.add_argument("--lpips_weight", type=float, default=0.9) # either 0.9 or 0.0, note that this must match the ckpt you use, because with weight>0, the lpips-vggnet weights were also saved during training. Incorrect state_dict keys may lead to load_state_dict error when loading the ckpt.
args = parser.parse_args()

# def save_image(x_recon, x, path, name):
#     img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
#     img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
#     img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
#     img = np.transpose(img, (1, 2, 0)).astype('uint8')
#     img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     img_final.save(os.path.join(path, name + '.png'))

from PIL import Image

def save_image(x_recon, x, path, name):
    img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
    img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
    img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
    img = np.transpose(img, (1, 2, 0)).astype('uint8')

    # Save img_recon
    img_recon_path = os.path.join(path, 'img_recon')
    if not os.path.exists(img_recon_path):
        os.makedirs(img_recon_path)
    img_recon_final = Image.fromarray(img_recon, 'RGB')
    img_recon_final.save(os.path.join(img_recon_path, name + '.png'))

    # Save img
    img_path = os.path.join(path, 'img')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    img_final = Image.fromarray(img, 'RGB')
    img_final.save(os.path.join(img_path, name + '.png'))


def lpipc(a: torch.Tensor, b: torch.Tensor) -> float:
    #lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips.LPIPS(net="alex")
    image1_tensor =  a.float() 
    image2_tensor =  b.float()
    device = torch.device("cuda")
    image1_tensor = image1_tensor.to(device)
    image2_tensor = image2_tensor.to(device)
    lpips_model = lpips_model.to(image1_tensor.device)
    distance = lpips_model(image1_tensor, image2_tensor)
    return distance


def load(model,ckpt, idx=0, load_step=True):
    data = torch.load(
            ckpt,
            map_location=torch.device(model.device)
            )
    if load_step:
        step = data["step"]
        model_to_load = model
    try:
        model_to_load.module.load_state_dict(data["model"], strict=False)
    except :
        model_to_load.load_state_dict(data["model"], strict=False)

def main(rank):
    path = args.img_dir
    resize = tuple([128, 256])
    test_dataset = PairCityscape(path=path, set_type='test', resize=resize)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=3,
        dim_mults=(1, 2, 3, 4, 5, 6),
        context_dim_mults=(1, 2, 3, 4),
    )
    # denoise_model_cor = Unet(
    #     dim=64,
    #     channels=3,
    #     context_channels=3,
    #     dim_mults=(1, 2, 3, 4, 5, 6),
    #     context_dim_mults=(1, 2, 3, 4),
    # )

    context_model = Compressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )
    context_model_cor = Compressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )
    context_w = Compressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )

    model = DICM(
        device=args.device,
        denoise_fn=denoise_model,
        # denoise_fn_cor=denoise_model_cor,
        context_fn=context_model,
        context_fn_cor=context_model_cor,
        context_w=context_w,
        num_timesteps=20000,
        clip_noise="none",
        vbr=False,
        lagrangian=0.9,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=args.lpips_weight,
        aux_loss_type="lpips"
    )



    # loaded_param = torch.load(
    #     args.ckpt,
    #     map_location=lambda storage, loc: storage,
    # )
    results_path = os.path.join(args.out_dir, 'test')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    names = ["Image Number", "BPP","tran-BPP", "PSNR", "MS-SSIM","DISTS","LPIPS"]
    cols = dict()


    load(model,args.ckpt)
    model.cuda()
    model.eval()
    mse = torch.nn.MSELoss(reduction='mean')
    mse = mse.cuda()
    D = DISTS().cuda()
    # inception_model = torchvision.models.inception_v3(pretrained=True)
    with torch.no_grad():
        for i, data in enumerate(iter(test_loader)):
        # img, _ ,_ = data
        # img = img.cuda().float()

            img, cor_img, _, _ = data
            img = img.float().cuda()
            cor_img = cor_img.float().cuda()

        # compressed_x, bpp= diffusion.compress(
        #     img * 2.0 - 1.0,
        #     sample_steps=200,
        #     sample_mode="ddim",
        #     bpp_return_mean=False,
        #     init=torch.randn_like(img) * 0.8
        # )

            compressed_x, compressed_y, bpp, transmitted_bpp = model.compress(
            img * 2.0 - 1.0,
            cor_img * 2.0 - 1.0,
            sample_steps=200,
            sample_mode="ddim",
            bpp_return_mean=False,
            init=torch.randn_like(img) * 0.8
            )

            x_recon = compressed_x.clamp(-1, 1) / 2.0 + 0.5

            dists_loss = D(img, x_recon, require_grad=False, batch_average=True)
            lp=lpipc(img,x_recon)
            mse_dist = mse(img, x_recon)
            msssim = 1 - ms_ssim(img.clone().cpu(), x_recon.clone().cpu(), data_range=1.0, size_average=True,
                             win_size=7)
            msssim_db = -10 * np.log10(msssim.item())

            vals = [str(i)] + ['{:.8f}'.format(x) for x in [bpp.item(),transmitted_bpp.item(),
                                                        10 * np.log10(1 / mse_dist.item()),
                                                        msssim_db.item(),dists_loss.item(),lp.item()]]


            for (name, val) in zip(names, vals):
                if name not in cols:
                    cols[name] = []
                cols[name].append(val)

            save_image(x_recon[0], img[0], os.path.join(results_path, '{}_images'.format('DICM')),
                       str(i))

    df = pd.DataFrame.from_dict(cols)
    df.to_csv(os.path.join(results_path, 'DICM' + '.csv'))


if __name__ == "__main__":
    main(args.device)
