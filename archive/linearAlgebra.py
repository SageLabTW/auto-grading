import os
import pdf2image as p2i
import numpy as np
import pandas as pd
# import pyzbar.pyzbar as zbar
from PIL import Image
import pickle


def extractor(path, output_folder = 'default', filename = 'default'):
    
    """
    Input:
        path: path of pdf file
        output_folder: name of output folder, default name is same as pdf file
        filename: name of jpg file, default name is {output_folder}_{}.jpg
    Output:
        None
    """
    
    
    # 建立資料夾，預設為 pdf 檔名
    if output_folder == 'default':
        tmp = path.split('/')
        tmp = tmp[-1].split('.')
        output_folder = tmp[0]
    try:
        os.mkdir(output_folder)
    except:
        print("Error: There exists a folder called \'{}\'.".format(output_folder))
        return
    
    # 定義每一張檔名，預設為"資料夾名稱_{}.jpg"
    if filename == 'default':
        filename = output_folder + '_{}.jpg'
    else:
        filename = filename + '_{}.jpg'
    output_path = output_folder + '/' + filename
    
    # 將 pdf 逐頁拆分並轉換成 jpg 並儲存
    imgs = p2i.convert_from_path(path, fmt = 'jpg', grayscale = True)
    for i, im in enumerate(imgs):
        im.save(output_path.format(i+1))
    
    n_img = len(imgs)
    checkcode = np.zeros([n_img,120,120])
    qrcode = np.zeros([n_img,400,400])
    for i, im in enumerate(imgs):
        
        # 擷取 checkcode 並記錄至 `checkcode`
        img = np.asarray(im)
        rect = img[img.shape[0]-400:,img.shape[1]-400:]    ### 選取最右下角的 400x400
        rect = rect.mean(axis = -1)
        edge_check = np.where(rect < 100)
        r = edge_check[0].max()
        c = edge_check[1].max()
        rect = rect[r-130:r-10, c-130:c-10]    ### -10 是從邊界往內縮，從右下角選取 120x120（checkcode 大小）
        checkcode[i,:,:] = rect
        
        # 擷取 QR code 並記錄至 `qrcode`
        qr = img[img.shape[0]-400:, :400]    ### 選取最左下角的 400x400
        qr = qr.mean(axis = -1)
        qrcode[i,:,:] = sharpen_vec(qr) ### 把黑色變得更黑
    
    # 將 `checkcode` 儲存，檔名預設為 checkcode_{output_folder}.pkl
    cc_name = 'checkcode_' + output_folder + '.pkl'
    with open(cc_name,'wb') as handle:
        pickle.dump(checkcode, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    # 將 `qrcode` 儲存，檔名預設為 qrcode{output_folder}.pkl
    qc_name = 'qrcode_' + output_folder + '.pkl'
    with open(qc_name,'wb') as handle:
        pickle.dump(qrcode, handle, protocol = pickle.HIGHEST_PROTOCOL)

    return



def make_sample(path, img_size = [32,32], filename = 'default'):
    # load checkcode
    with open(path, 'rb') as handle:
        checkcode = pickle.load(handle)
    # resize checkcode and record to `sample`
    sample = np.zeros((checkcode.shape[0], img_size[0], img_size[1]))
    for i, rect in enumerate(checkcode):
        rect = Image.fromarray(rect)
        rect = rect.resize((img_size[0],img_size[1]), Image.ANTIALIAS)    ### resize 成新的大小，預設為 32
        rect = np.asarray(rect)
        rect = (rect - rect.min())/(rect.max() - rect.min()) ### normalize to 0~1
        sample[i,:,:] = rect
    
#     # save sample，檔名預設為 sample_{}.pkl
#     if filename == 'default':
#         tmp = path.split('/')
#         tmp = tmp[-1].split('.')
#         tmp = tmp[0].split('_')
#         filename = 'sample_' + tmp[-1] + '.pkl'
#     else:
#         filename = filename + '.pkl'
#     with open(filename, 'wb') as handle:
#         pickle.dump(checkcode, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    # return `sample`
    return sample




def detect(I): 
    # 偵測所有的 QR code
    barcodes = zbar.decode(I)
    
    # 逐一解碼，回傳位置與結果
    bbox = []; msg = [];
    for i, barcode in enumerate(barcodes):
        bbox.append(np.array(barcode.rect))
        msg.append(barcode.data.decode('utf-8'))

    return msg



def sharpen(k):
    if k > 180: ### 180 是測試出來的
        return 255
    else:
        return 0



### 將函數改成 numpy 可用的函數
def sharpen_vec(k):
    f = np.vectorize(sharpen)
    return f(k)

# 定義完這兩個以後
# 原本的程式可以更改三行
# 成功率可以到 100% （以目前的資料來說）