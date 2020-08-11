import os
import pdf2image as p2i
import numpy as np
import pandas as pd
import pyzbar.pyzbar as zbar
from PIL import Image
import pickle



def extractor(path, output_folder = 'default', filename = 'default', save_checkcode = True, save_qrcode = True):
    
    """
    Input:
        path: path of pdf file.
        output_folder: name of output folder, default name is same as pdf file.
        filename: name of jpg file, default name is {output_folder}_{}.jpg.
        save_checkcode: save checkcode in numpy array format as .pkl file. Default is True.
        save_qrcode: save qrcode in numpy array format as .pkl file. Default is True.
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
    
#     # 將 pdf 逐頁拆分並轉換成 jpg
    imgs = p2i.convert_from_path(path, fmt = 'jpg', grayscale = True)
    
    n_img = len(imgs)
    checkcode = np.zeros([n_img,120,120])
    qrcode = np.zeros([n_img,400,400])
    for i, im in enumerate(imgs):
        # 擷取 checkcode
        img = np.asarray(im)
        rect = img[img.shape[0]-350:,img.shape[1]-350:]    ### 選取最右下角的 350x350
        rect_gray = rect.mean(axis = -1)
        r, c = find_edge(rect)
        rect_gray = rect_gray[r-130:r-10, c-130:c-10]    ### -10 是從邊界往內縮，從右下角選取 120x120（checkcode 大小）
        checkcode[i,:,:] = rect_gray
        rect = rect[r-130:r-10, c-130:c-10]
        rect = Image.fromarray(rect)
        rect.save(output_path.format(i+1))    ### 將每一張截取的 checkcode 儲存為 jpg file
        
        # 擷取 QR code
        qr = img[img.shape[0]-400:, :400]    ### 選取最左下角的 400x400
        qr = qr.mean(axis = -1)
        qrcode[i,:,:] = sharpen_vec(qr) ### 把黑色變得更黑
    
    # 將 checkcode 儲存，檔名預設為 checkcode_{output_folder}.pkl
    if save_checkcode:
        cc_name = 'checkcode_' + output_folder + '.pkl'
        with open(cc_name,'wb') as handle:
            pickle.dump(checkcode, handle, protocol = pickle.HIGHEST_PROTOCOL)
    else:
        pass
    
    # 將 qrcode 儲存，檔名預設為 qrcode{output_folder}.pkl
    if save_qrcode:
        qc_name = 'qrcode_' + output_folder + '.pkl'
        with open(qc_name,'wb') as handle:
            pickle.dump(qrcode, handle, protocol = pickle.HIGHEST_PROTOCOL)
    else:
        pass
    return


# extractor 裡所需的函數，功能為尋找 checkcode 邊界
def find_edge(rect, threshold = 180):
    rect_gray = rect.mean(axis = -1)
    edge_check = np.where(rect_gray < threshold)
    r = edge_check[0].max()
    c = edge_check[1].max()
    row_cond = np.sum(rect_gray[r-3:r+1,:] < threshold) >= 300
    col_cond = np.sum(rect_gray[:,c-3:c+1] < threshold) >= 300
    if row_cond:
        pass
    else:
        tmp = np.sort(edge_check[0])
        i = 1
        while True:
            r = tmp[-1 - i]
            row_cond = np.sum(rect_gray[r-3:r+1,:] < threshold) >= 300
            if row_cond:
                break
            else:
                i += 1
    if col_cond:
        pass
    else:
        tmp = np.sort(edge_check[1])
        i = 1
        while True:
            c = tmp[-1 - i]
            col_cond = np.sum(rect_gray[:,c-3:c+1] < threshold) >= 300
            if col_cond:
                break
            else:
                i += 1
    return r, c

    


# 建立 train data
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



# 讀取 qrcode 裡的訊息並回傳一個 pandas Series
def qr_decoder(qrcode_path, key_path, filename = 'default'):
    # load qrcode
    with open(qrcode_path, 'rb') as handle:
        qrcode = pickle.load(handle)
    
    # load key
    key = pd.read_csv(key_path)
    
    # decode and record to `msg`
    msg = pd.Series(np.zeros(qrcode.shape[0], dtype = int))
    for i, img in enumerate(qrcode):
        try:
            tmp_msg = detect(img)
            tmp_msg = tmp_msg[0][1:]
            msg[i] = key.CHECKCODE.loc[key.KEY == tmp_msg].values[0]
        except:
            print(i)
            msg[i] = -1    ### 沒有掃描到 ＱＲcode
    
    # return `msg`
    return msg



# qr_decoder 裡所需的函數，功能為讀取 qrcode 裡的訊息
def detect(I): 
    # 偵測所有的 QR code
    barcodes = zbar.decode(I)
    
    # 逐一解碼，回傳位置與結果
    bbox = []; msg = [];
    for i, barcode in enumerate(barcodes):
        bbox.append(np.array(barcode.rect))
        msg.append(barcode.data.decode('utf-8'))

    return msg


# qr_decoder 裡所需的函數，功能為增加 qrcode 對比度
def sharpen(k):
    if k > 180: ### 180 是測試出來的
        return 255
    else:
        return 0



# 將 sharpen 改成 numpy 可用的函數
def sharpen_vec(k):
    f = np.vectorize(sharpen)
    return f(k)

# 定義完這兩個以後
# 原本的程式可以更改三行
# 成功率可以到 100% （以目前的資料來說）