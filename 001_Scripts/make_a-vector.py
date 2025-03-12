import argparse, os
import PIL
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import pickle


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

# model setting
    path= '/home/jinny/projects/Art-history/Art-history/'
    config = OmegaConf.load(path+f"scripts/configs/stable-diffusion/v2-inference.yaml")
    model = load_model_from_config(config, path+f"scripts/configs/checkpoint/512-base-ema.ckpt") 
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

#Path 불러오기
file_info = pd.read_csv('/home/jinny/projects/Art-history/Art-history/datas/file_info.csv')

# model 돌리기
latent_fail = []
image_latent = []
for i in file_info.Path :
    try :
        init_image = load_img('/home/jinny/projects/Art-history/Art-history/datas/resized_image/'+i).to(device)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        init_latent = init_latent.view([-1])
        image_latent.append([init_latent.cpu().numpy(),i])
    except Exception as e:
        print(e)
        latent_fail.append(i)
        continue

np.save('/home/jinny/projects/Art-history/Art-history/datas/fail_paths', np.array(latent_fail))
np.save('/home/jinny/projects/Art-history/Art-history/datas/vectors/avec_latents', np.array(image_latent))