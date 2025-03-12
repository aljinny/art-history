import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.image import imread

import cv2
import pickle

from wordcloud import WordCloud
from collections import Counter
import re

base_path= '/home/jinny/projects/Art-history/Art-history/'

prompts = pd.read_csv(base_path+'datas/prompts.csv')
file_info = pd.read_csv(base_path+'datas/file_info.csv')
avetor = np.load(base_path+'datas/vectors/avec_latents.npy', allow_pickle=True)

# file_info_latent
df = pd.DataFrame((avetor),columns=['latent','Path']) 
file_info_latents = pd.merge(file_info, df, how = 'left', on = 'Path')
file_info_latents = file_info_latents[~file_info_latents.latent.isnull()]
file_info_latents = pd.merge(file_info_latents, prompts, how = 'left', on = 'Path')

import os

import argparse, os
import PIL
import torch
import numpy as np
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
import pandas as pd
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.image import imread

import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import umap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.image import imread

from PIL import Image,ImageOps

from sklearn.decomposition import PCA

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import torch
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

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

def load_model(device_num) :
    config = OmegaConf.load(base_path+f"scripts/configs/stable-diffusion/v2-inference.yaml")
    model = load_model_from_config(config, base_path+f"scripts/configs/checkpoint/512-base-ema.ckpt") 

    GPU_NUM = device_num 

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
    
    model = model.to(device)
    return model,device

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

def model_2img(latent_samples, model) :
    precision_scope = autocast 
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                
                x_samples = model.decode_first_stage(latent_samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    return img


def run_diffusion_model_changing_denoising_steps(init_latent,prompt,n,num,batch,model,device,step):  

    seed_everything(980727)
    
    ddim_steps = 50
    ddim_eta = 0.0
    n_iter = 1
    batch_size = batch
    scale = 9.0
    
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    precision_scope = autocast
    
    all_samples = []
    all_samples_enc = []
    for i in range(n):
        strength = 1/ddim_steps * (i+num) # change strength to control denoising steps
        data = [batch_size * [prompt]]
        t_enc = step
        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in trange(n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            if prompt == "no prompt" :
                                print("change prompt vector to zero")
                                c = torch.zeros(c.shape[0], c.shape[1], c.shape[2]).to('cuda:0') # change prompt vector to zero (i.e., no direction)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc, )

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            
        all_samples.append(x_samples)
        all_samples_enc.append(samples)
               
    return all_samples_enc

# model load
model,device = load_model(0)

with open(base_path+'datas/vectors/diffusion/diffusion_sample.pkl', 'rb') as file:
    paintings = pickle.load(file)

### diffusion prompt no sep
result_5 = dict()
result_10 = dict()
result_20 = dict()
result_30 = dict()

for i in range(4) :
    year = 1500+100*i
    temp5 = list()
    temp10 = list()
    temp20 = list()
    temp30 = list()
    
    for idx,res in enumerate(paintings[year]):
        print(year,'----',idx)
        batch = 1
        res = res[0].reshape(1,4,64,64)
        
        result = torch.tensor(res)
        result = result.to(device)   
        result = run_diffusion_model_changing_denoising_steps(result,'',1,1,batch,model,device,5)
        result = result[0].reshape(1,4,64,64)
        temp5.append(result)
            
        result = torch.tensor(res)
        result = result.to(device)   
        result = run_diffusion_model_changing_denoising_steps(result,'',1,1,batch,model,device,10)
        result = result[0].reshape(1,4,64,64)
        temp10.append(result)

        result = torch.tensor(res)
        result = result.to(device)   
        result = run_diffusion_model_changing_denoising_steps(result,'',1,1,batch,model,device,20)
        result = result[0].reshape(1,4,64,64)
        temp20.append(result)

        result = torch.tensor(res)
        result = result.to(device)   
        result = run_diffusion_model_changing_denoising_steps(result,'',1,1,batch,model,device,30)
        result = result[0].reshape(1,4,64,64)
        temp30.append(result)
        
    result_5[year] = temp5
    result_10[year] = temp10
    result_20[year] = temp20
    result_30[year] = temp30

with open(base_path+'datas/vectors/diffusion/no_comma/avec/step5/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_5, file)

with open(base_path+'datas/vectors/diffusion/no_comma/avec/step10/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_10, file)

with open(base_path+'datas/vectors/diffusion/no_comma/avec/step20/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_20, file)

with open(base_path+'datas/vectors/diffusion/no_comma/avec/step30/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_30, file)

with open(base_path+'datas/vectors/diffusion/comma/avec/step5/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_5, file)

with open(base_path+'datas/vectors/diffusion/comma/avec/step10/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_10, file)

with open(base_path+'datas/vectors/diffusion/comma/avec/step20/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_20, file)

with open(base_path+'datas/vectors/diffusion/comma/avec/step30/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_30, file)


for year in tqdm([1500,1600,1700,1800]) :
    result_5[year] = [model_2img(i, model) for i in result_5[year] ]

for year in tqdm([1500,1600,1700,1800]) :
    result_10[year] = [model_2img(i, model) for i in result_10[year] ]

for year in tqdm([1500,1600,1700,1800]) :
    result_20[year] = [model_2img(i, model) for i in result_20[year] ]

for year in tqdm([1500,1600,1700,1800]) :
    result_30[year] = [model_2img(i, model) for i in result_30[year] ]


with open(base_path+'datas/vectors/diffusion/no_comma/img/step5/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_5, file)

with open(base_path+'datas/vectors/diffusion/no_comma/img/step10/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_10, file)

with open(base_path+'datas/vectors/diffusion/no_comma/img/step20/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_20, file)

with open(base_path+'datas/vectors/diffusion/no_comma/img/step30/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_30, file)

with open(base_path+'datas/vectors/diffusion/comma/img/step5/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_5, file)

with open(base_path+'datas/vectors/diffusion/comma/img/step10/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_10, file)

with open(base_path+'datas/vectors/diffusion/comma/img/step20/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_20, file)

with open(base_path+'datas/vectors/diffusion/comma/img/step30/using_noprompt.pkl', 'wb') as file:
    pickle.dump(result_30, file)