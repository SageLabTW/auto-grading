import extractor as ex
import annotator as an

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

np.set_printoptions(precision=2)



def shift_n_scale(arr, shift, scale=1):
    p,q = shift
    pad = max(abs(p), abs(q))
    a,b = arr.shape
    new = np.zeros((a + 2*pad, b + 2*pad), dtype=arr.dtype)
    new[pad+p:pad+p+a, pad+q:pad+q+b] = scale*arr
    return new[pad:pad+a, pad:pad+b]

def thicken(arr, rad=3, drop=0.8):
    moves = []
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            dist = abs(i) + abs(j)
            if dist <= rad - 1:
                moves.append(shift_n_scale(arr, (i,j), scale=drop**dist))
    new_arr = np.array(moves)
    return new_arr.max(axis=0)

def img2arr(path, out_range=16, size=(8,8), rev=True):
    img = Image.open(path).resize(size)
    img_arr = np.array(img)
    if rev:
        img_arr = img_arr.max() - img_arr 
    in_range = float(img_arr.max() - img_arr.min())
    if in_range == 0:
#         print("Brightness is uniform")
        msg = "Brightness is uniform"
    else:
        img_arr = img_arr / in_range * out_range
        msg = None
#         img_arr = img_arr.astype(int)
    return img_arr, msg
        
def imgs2arr(raw=None, ext='png', out_range=16, size=(8,8), rev=True, rad=1, drop=0.8, level=None):
    if raw == None:
        raw = ex.raw_data('nsysu-digits')
    files = [f for f in raw.df[0]]
    a,b = len(files),size[0]*size[1]
    arr = np.zeros((a,b), dtype=int)
    for i,f in enumerate(files):
        img_arr, msg = img2arr(os.path.join(raw.path,f), 
                          out_range=out_range, 
                          size=size, 
                          rev=rev)
        if msg != None:
            print(f + ': ' + msg)
        if rad > 1:
            arr[i] = thicken(img_arr, rad=rad, drop=drop).reshape(b)
        else:
            arr[i] = img_arr.reshape(b)
    if level != None: # level=(50,255,200,255)
        mask = (arr>=level[0]) & (arr<=level[1])
        ratio = (level[3] - level[2]) / (level[1] - level[0])
        arr[mask] = (arr[mask] - level[0])*ratio + level[2]
    return arr

def labels(raw=None):
    if raw == None:
        raw = ex.raw_data('nsysu-digits')
    return raw.df[1].values.astype(int)

def show(imgs, ans, size=None):
    fig,axs = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            axs[i,j].axis('off')
            if size != None:
                img = imgs[5*i+j].reshape(*size)
            else:
                img = imgs[5*i+j]
            axs[i,j].imshow(img, cmap="Greys")
            axs[i,j].set_title(ans[5*i+j])
    plt.show()
    plt.close()
    
def predict(mdl_path, data_path):
    ### load model and data
    mdl = tf.keras.models.load_model('OCR_mdl.h5')
    data = ex.raw_data(data_path)
    X_test = imgs2arr(data, size=(28,28))
    X_test = X_test.reshape(X_test.shape[0],28,28,1)
    
    ### predict
    y_pred = mdl.predict(X_test).argmax(axis=1)
    
    ### save result as a csv file
    df_new = data.df.copy()
    df_new[1] = y_pred
    df_new.to_csv(os.path.join(data_path, data_path+'_pred.csv'),
                  header=False,
                  index=False)














    