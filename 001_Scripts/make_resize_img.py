import numpy as np
import pandas as pd
from collections import Counter
import re
import cv2
from tqdm import tqdm
import os

file_info = pd.read_csv('/home/jinny/projects/Art-history/Art-history/datas/file_info.csv')

def create_directory(outDir) :

    if not os.path.exists('/home/jinny/projects/Art-history/Art-history/datas/resized_image/'+outDir[0]):
        os.mkdir('/home/jinny/projects/Art-history/Art-history/datas/resized_image/'+outDir[0])
    if not os.path.exists('/home/jinny/projects/Art-history/Art-history/datas/resized_image/'+outDir[0]+'/'+outDir[1]):
        os.mkdir('/home/jinny/projects/Art-history/Art-history/datas/resized_image/'+outDir[0]+'/'+outDir[1])

def resize_cv2(image, target_size=(512, 512)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
def process_image(imgFile, outDir):
    
    # Create output directory
    outDir=outDir.split('/')
    create_directory(outDir)
    # Resize
    image = cv2.imread(imgFile)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_image = resize_cv2(image, (512, 512))
    cv2.imwrite(f'/home/jinny/projects/Art-history/Art-history/datas/resized_image/{outDir[0]}/{outDir[1]}/{outDir[2]}', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))

# Run
resize_fail = []
for i in tqdm(file_info['Path']) :
    try :
        process_image( imgFile='/home/jinny/datas/art500k/'+i, outDir=i)
    except Exception as e :
        resize_fail.append(i)
        print(e)
        continue
np.save('resize_fail', np.array(resize_fail))