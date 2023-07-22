import extractor as ex
import annotator as an

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from PIL import Image

import warnings

np.set_printoptions(precision=2)


### NORMALIZATION
def shift_n_scale(arr, shift, scale=1):
    p,q = shift
    pad = max(abs(p), abs(q))
    a,b = arr.shape
    new = np.zeros((a + 2*pad, b + 2*pad), dtype=arr.dtype)
    new[pad+p:pad+p+a, pad+q:pad+q+b] = scale*arr
    return new[pad:pad+a, pad:pad+b]

def thicken(arr, rad=2, decay=0.8):
    moves = []
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            dist = abs(i) + abs(j)
            if dist <= rad - 1:
                moves.append(shift_n_scale(arr, (i,j), scale=decay**dist))
    new_arr = np.array(moves)
    return new_arr.max(axis=0)

def level(arr, thres=10, a=1, b=100):
    m,n = arr.shape
    new_arr = arr.copy()
    new_arr[arr > thres] = new_arr[arr > thres] * a + b
    upd = np.zeros_like(arr) + 255
    thick = np.concatenate([new_arr[np.newaxis,:,:], upd[np.newaxis,:,:]], axis=0)
    return thick.min(axis=0)

def bounding_box(arr, thres=10, out='subarray'):
    """
    out can be 'bounds' or 'subarray'
    """
    xs,ys = np.where(arr > 10)
    if out == 'bounds':
        return xs.min(), xs.max(), ys.min(), ys.max()
    if out == 'subarray':
        return arr[xs.min():xs.max()+1, ys.min():ys.max()+1]
    
def out_size(in_size, target=20):
    x,y = in_size
    big = max(x,y)
    ratio = float(target) / big 
    out_x = target if x == big else int(np.ceil(x*ratio))
    out_y = target if y == big else int(np.ceil(y*ratio))
    return (out_x, out_y)

def arr_centers(arr):
    m,n = arr.shape
    row_sum = np.sum(arr, axis=1)
    v_cen = (row_sum * np.arange(m)).sum() / row_sum.sum()
    col_sum = np.sum(arr, axis=0)
    h_cen = (col_sum * np.arange(n)).sum() / col_sum.sum()
    return (v_cen, h_cen)

def centerize(arr, thres=10, target=20):
    if (arr < thres).all():
        warnings.warn("Found a blank picture.")
        return arr 
    
    m,n = arr.shape
    new_arr = np.zeros((m + 2*target, n + 2*target), dtype=arr.dtype)
    
    img = Image.fromarray(bounding_box(arr, thres=thres).astype('uint8'))
    o_size = out_size(img.size, target=target)   
    re_arr = np.array(img.resize(o_size), dtype=arr.dtype)
    v_cen,h_cen = arr_centers(re_arr)
    v = target + int(np.round(0.5*n - v_cen))
    h = target + int(np.round(0.5*m - h_cen))
    vp = v + re_arr.shape[0]
    hp = h + re_arr.shape[1]
    
    new_arr[v:vp, h:hp] = re_arr
    return new_arr[target:target+m, target:target+n]

def normalize(X):
    """
    X is an array of flattened 28x28 images.
    This function will make changes of X directly.
    This function requires the images to be 0: white and 255: black.
    """
    for i in range(X.shape[0]):
        arr = X[i].reshape(28,28)
        arr = thicken(arr) ### decide whether to thicken
        arr = level(arr) ### decide whether to darken
        arr = centerize(arr) ### decide whether to center
        X[i] = arr.reshape(784)   
    return None

def rev(X, out_range=255):
    """
    flip black and white
    """
    X *= -1
    X += out_range
    return None

### LOADING
def img2arr(path, out_range=255, size=(28,28)):
    img = Image.open(path).resize(size)
    img_arr = np.array(img)
#     if rev:
#         img_arr = img_arr.max() - img_arr 
    in_range = float(img_arr.max())
    if in_range == 0:
#         print("Brightness is uniform")
        msg = "Brightness is uniform"
    else:
        img_arr = img_arr / in_range * out_range
        msg = None
#         img_arr = img_arr.astype(int)
    return img_arr, msg
        
def imgs2arr(raw, out_range=255, size=(28,28)):
    files = [f for f in raw.df[0]]
    a, b = len(files), size[0] * size[1]
    arr = np.zeros((a,b), dtype=int)
    for i,f in enumerate(files):
        img_arr, msg = img2arr(os.path.join(raw.path,f), 
                          out_range=out_range, 
                          size=size)
        if msg != None:
            print(f + ': ' + msg)
        arr[i] = img_arr.reshape(-1)
#         if rad > 1:
#             arr[i] = thicken(img_arr, rad=rad, drop=drop).reshape(b)
#         else:
#             arr[i] = img_arr.reshape(b)
#     if level != None: # level=(50,255,200,255)
#         mask = (arr>=level[0]) & (arr<=level[1])
#         ratio = (level[3] - level[2]) / (level[1] - level[0])
#         arr[mask] = (arr[mask] - level[0])*ratio + level[2]
    return arr

def labels(raw):
    return raw.df[1].values.astype(int)

def show(imgs, ans=None, size=None):
    if ans==None:
        ans = -np.ones(imgs.shape[0])
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
    
def predict(raw, mdl_path="svc-1639.joblib", normalize_data=True):
    """
    available models: OCR_mdl.h5, svc-1639.joblib
    """
    ### load model and data
    ext = mdl_path.split('.')[-1]
    if ext == 'h5':
        loaded_model = tf.keras.models.load_model(mdl_path)
    if ext == 'joblib':
        loaded_model = joblib.load(mdl_path)
        
    ### load data
    X = imgs2arr(raw)
    rev(X)
    if normalize_data:
        normalize(X)
    
    if ext == 'h5': ### tensorflow CNN input format
        X = X.reshape(raw.num, 28, 28, 1)
    if ext == 'joblib': ### sklearn input format
        X = X.reshape(raw.num, -1)
    
    ### predict
    if ext == 'h5': ### tensorflow CNN input format
        y_pred = loaded_model.predict(X).argmax(axis=1)
    if ext == 'joblib': ### sklearn input format
        y_pred = loaded_model.predict(X)
    
    ### save result as a csv file
    df_new = raw.df.copy()
    df_new[1] = y_pred
    df_new[2] = 'r'
    df_new.to_csv(os.path.join(raw.path, raw.path+'_pred.csv'),
                  header=False,
                  index=False)

def make_csv():
    ### only for nsysu-digits so far
    raw = ex.raw_data('nsysu-digits')
    X = imgs2arr(raw)
    rev(X) # 0: white, 255: black
    y = labels(raw)
    os.chdir('nsysu-digits')
    np.savetxt("X.csv", X, fmt='%d', delimiter=',')
    np.savetxt("y.csv", y, fmt='%d', delimiter=',')
    os.chdir('..')













    