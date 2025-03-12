import pandas as pd
import numpy as np
from PIL import Image
from clip_interrogator import Config, Interrogator
import sys
from tqdm import tqdm

def main() :
    print(sys.argv[1])
    
    # load dataset
    file_info = pd.read_csv('/home/jinny/projects/Art-history/Art-history/datas/file_info.csv')
    
    # load clip
    ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k",device='cuda:0'))
    
    if sys.argv[1]=='test' :
        file_prompts = list()
        file_prompt_vec = list()
        for path in tqdm(file_info['Path'][:1]) :
            image = Image.open(f'/home/jinny/datas/art500k/{path}').convert('RGB')
            file_prompts.append(np.array([ci.interrogate(image),path]))
            file_prompt_vec.append(np.array([np.array(ci.image_to_features(image).cpu()),path]))
    else :
        # make prompts
        file_prompts = list()
        file_prompt_vec = list()
        for idx,path in enumerate(file_info['Path'][0+12750*int(sys.argv[1]):12750+12750*int(sys.argv[1])]) :
            print(f'----------------------{idx}----------------------')
            try :
                image = Image.open(f'/home/jinny/datas/art500k/{path}').convert('RGB')
                file_prompts.append(np.array([ci.interrogate(image),path]))
                file_prompt_vec.append(np.array([np.array(ci.image_to_features(image).cpu()),path]))
            except :
                fail_images.append(path)
        
    np.save(f'/home/jinny/projects/Art-history/Art-history/datas/vectors/prompts_0{sys.argv[1]}', np.array(file_prompts))
    np.save(f'/home/jinny/projects/Art-history/Art-history/datas/vectors/cvec_latents_0{sys.argv[1]}', np.array(file_prompt_vec))

if __name__ == '__main__' :
    main()