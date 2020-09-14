
import os
import shutil
import pdf2image as p2i
from PIL import ImageDraw
import pyzbar.pyzbar as zbar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean(folder):
    """rm -rf folder"""
    shutil.rmtree(folder)

boxes = {"1081f": None, 
         "1081_1": (1353,2022,1479,2148), # tested
         "1081_2": (1353,2022,1479,2148), # not tested
         "1081_3": (1353,2022,1479,2148), # not tested
         "1082_1": (1377,2047,1503,2173), # tested
         "1082_2": (1377,2047,1503,2173), # not tested
         "1082_3": (1377,2047,1503,2173), # not tested
         "a4": (1417.78, 2102.78, 1575.26, 2260.26) # not tested
        }


def find_box(img, threshold=180, box_size=120):
    img_arr = np.asarray(img)
    rect = img_arr[img_arr.shape[0]-350:,img_arr.shape[1]-350:]
    edge_check = np.where(rect < threshold)
    r = edge_check[0].max()
    c = edge_check[1].max()
    row_cond = np.sum(rect[r-3:r+1,:] < threshold) >= 300
    col_cond = np.sum(rect[:,c-3:c+1] < threshold) >= 300
    if row_cond:
        pass
    else:
        tmp = np.sort(edge_check[0])
        i = 1
        while True:
            r = tmp[-1 - i]
            row_cond = np.sum(rect[r-3:r+1,:] < threshold) >= 300
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
            col_cond = np.sum(rect[:,c-3:c+1] < threshold) >= 300
            if col_cond:
                break
            else:
                i += 1
    r = img_arr.shape[0] - 350 + r
    c = img_arr.shape[1] - 350 + c
    box = (c-10-box_size, r-10-box_size, c-10, r-10)
    return box


# extract 中 extract key 所需的函數，功能為讀取 qrcode 裡的訊息
def detect(I): 
    # 偵測所有的 QR code
    barcodes = zbar.decode(I)
    
    # 逐一解碼，回傳位置與結果
    msg = [];
    for i, barcode in enumerate(barcodes):
        msg.append(barcode.data.decode('utf-8'))
    return msg


# qr_fixer 裡所需的函數，功能為增加 qrcode 對比度
def sharpen(k):
    if k > 180: ### 180 是測試出來的
        return 255
    else:
        return 0
sharpen_vec = np.vectorize(sharpen)


# extract 中 extract key 所需的函數，功能為找出ＱＲcode所在的位置
def find_qr_box(img, box_size=160):
    num_threshold = 150
    img = sharpen_vec(img)
    rect = img[img.shape[0]-400:,:400]
    edge_check = np.where(rect == 0)
    r = edge_check[0].max()
    c = edge_check[1].min()
    row_cond = np.sum(rect[r-3:r+1,:] == 0) >= num_threshold
    col_cond = np.sum(rect[:,c-3:c+1] == 0) >= num_threshold
    if row_cond:
        pass
    else:
        tmp = np.sort(edge_check[0])
        i = 1
        while True:
            r = tmp[-1 - i]
            row_cond = np.sum(rect[r-3:r+1,:] == 0) >= num_threshold
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
            c = tmp[i]
            col_cond = np.sum(rect[:,c-3:c+1] == 0) >= num_threshold
            if col_cond:
                break
            else:
                i += 1
    r = img.shape[0] - 400 + r
    qr_box = (c-5, r-box_size+5, c+box_size-5, r+5, )
    return qr_box


# extract 中 extract key 所需的函數，功能為增加ＱＲcode辨識率
def qr_fixer(img):
    img = sharpen_vec(img)
    idx = np.where(img == 255)
    for i in range(len(idx[0])):
        r_idx = idx[0][i]
        c_idx = idx[1][i]
        tmp_rect = img[r_idx-1:r_idx+2,c_idx-1:c_idx+2]
        if np.sum(tmp_rect) <= 255*4:
            img[r_idx,c_idx] = 0
    return img


    
def extract(path, key_path, box='auto', get_key=False, output_folder='default', filename='default', pages=None, test_mode=False, cc_box_size=120, qr_box_size=160):
    """Write png files and a csv file to output_folder.
    
    Input:
        path: path of pdf file
        output_folder: name of output folder, default name is same as pdf file
        filename: suffix of output filename, default name is {output_folder}_{}.pgm
        box: (left, top, right, bottom) for cropping checkcode
            108 fake: ???
            1081: (1353,2022,1479,2148)
            1082: (1377,2047,1503,2173)
            a4: (1417.78, 2102.78, 1575.26, 2260.26) not tested
    Note: 
        a4 page size = 29.7 x 21 cm; 11.69 x 8.27 in; 2339 x 1654 in 200 dpi
        1cm = 0.3937 inches = 78.74 pixels in 200 dpi
    Output:
        None
    """
    
    
    # 建立資料夾，預設為 pdf 檔名
    if output_folder == 'default':
        fn = os.path.split(path)[-1]
        output_folder = os.path.splitext(fn)[0] 
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        print("Error: There exists a folder called \'{}\'.".format(output_folder))
        question = "Do you want to remove the folder? [y/N]"
        ans = input(prompt=question)
        if ans in ['y', 'Y']:
            clean(output_folder)
            os.mkdir(output_folder)
        elif ans in ['n', 'N', '', None]:
            return
    
    # 定義每一張檔名，預設為"資料夾名稱_{}.jpg"
    if filename == 'default':
        filename = output_folder + '_{:03d}.png'
        filename_paper = 'paper_' + output_folder + '_{:03d}.png'
    else:
        filename = filename + '_{:03d}.png'
        filename_paper = 'paper_' + filename + '_{:03d}.png'
    output_path = os.path.join(output_folder, filename)
    output_path_paper = os.path.join(output_folder, filename_paper)
    
    # 將 pdf 逐頁拆分並轉換成 png 並儲存
    # if test_mode is on; use color photos
    if pages != None:
        imgs = p2i.convert_from_path(path, grayscale = not test_mode, 
                                     first_page=pages[0], last_page=pages[1])
    else:
        imgs = p2i.convert_from_path(path, grayscale = not test_mode)
    
    ### create csv
    num_imgs = len(imgs)
    df = pd.DataFrame([
            [filename.format(i) for i in range(num_imgs)], 
            [None]*num_imgs
        ])
    df.T.to_csv(os.path.join(output_folder, output_folder+'.csv'), 
                header=False, 
                index=False)
    
    # prepare for extract key
    if get_key:
        key = pd.read_csv(key_path)    ### load key
        msg = pd.Series(np.zeros(num_imgs, dtype = int))    ### for recording key
    
    
    auto_boxing = True if box == 'auto' else False
    for i, im in enumerate(imgs):
        ### save paper
        im.save(output_path_paper.format(i))
        
        ### find box
        if auto_boxing:
            box = find_box(im, box_size=cc_box_size)
            
        ### extract checkcode
        if test_mode:
            w = box[2] - box[0]
            h = box[3] - box[1]
            r = 0.3 ### extra padding
            view_box = (box[0] - r*w, box[1] - r*h, box[2] + r*w, box[3] + r*h)
            draw = ImageDraw.Draw(im)
            draw.rectangle(box, width=3, outline='red')
            checkcode = im.crop(view_box)
            
        else:
            checkcode = im.crop(box)
        checkcode = checkcode.resize((28,28))
        checkcode.save(output_path.format(i))
        
        ### extract key
        if get_key:
            qr_box = find_qr_box(im, box_size = qr_box_size)
            qrcode = im.crop(qr_box)
            tmp_msg = detect(qrcode)
            try:
                tmp_msg = tmp_msg[0][1:]
                msg[i] = key.CHECKCODE.loc[key.KEY == tmp_msg].values[0]
            except:
                qrcode = qr_fixer(qrcode)
                tmp_msg = detect(qrcode)
                try:
                    tmp_msg = tmp_msg[0][1:]
                    msg[i] = key.CHECKCODE.loc[key.KEY == tmp_msg].values[0]
                except:
                    print('get qrcode failed: {}'.format(filename.format(i)))
                    msg[i] = -1    ### 沒有掃描到 ＱＲcode

        ### TBD
    
    # save qr.csv
    if get_key:
        df = df.T
        df[1] = msg
        df.to_csv(os.path.join(output_folder, output_folder+'_qr.csv'),
                  header=False,
                  index=False)
        
class raw_data:
    def __init__(self, path):
        """initiate the object with path = folder_name"""
        self.path = path
        self.name = os.path.split(path)[-1]
        self.df = pd.read_csv(os.path.join(path, self.name+'.csv'), 
                              header=None)
        self.num = self.df.shape[0]
    
    def examine(self, start=None, end=None, each_row=5, size=2, label='data'):
        if start == None:
            start = 0
        if end == None:
            end = self.num
        if isinstance(label, str) and label == 'data':
            label = self.df

        total = end - start
        rows = total // each_row
        if total % each_row != 0:
            rows += 1
            
        fig,axs = plt.subplots(rows, each_row, 
                               figsize=(size*each_row,size*rows),
                               squeeze=False)

        for k in range(total):
            real_k = start + k
            i,j = k//each_row, k%each_row
            fn = label.iloc[real_k,0]
            ax = axs[i][j]
            ax.axis('off')
            img = plt.imread(os.path.join(self.path, fn))
            ax.imshow(img, cmap='Greys_r', vmin=0, vmax=1)
            ax.set_title('%s'%label.iloc[real_k,1])
            ax.text(0, 27, fn)
        plt.show()
        
        return fig
    
    def labeler(self, start=None, end=None, each_row=5, size=2, patch=None):
        """API for examining and modifying the data and labels"""
        if start == None:
            start = 0
        if end == None:
            end = self.num

        total = end - start
        rows = total // each_row
        if total % each_row != 0:
            rows += 1
            
        changes = []
        if patch != None:
            new_df = pd.read_csv(patch, 
                                 header=None
                                ).rename({2:'notes'}, axis=1)
            print("You indicated the patch %s."%patch)
            review = input(prompt="Do you want to review the changes? [Y/n]")
            if review in ['y', 'Y', '']:
                review = True
            elif review in ['n', 'N']:
                review = False
        else:
            new_df = self.df.copy()
            new_df['notes'] = ''
            review = True
        i = 0
        
        while True:
            if review:
                if i >= rows:
                    print('Reaching the end.  [s]ave or [q]uit?')
                else:
                    self.examine(start + i*each_row, 
                                 min(start + (i+1)*each_row, total), 
                                 each_row, size, 
                                 label=new_df)

                    print('Give me five digits for your changes: [h] help')

                c = input()
            else:
                c = 's'
                
            if c == 'h':
                print("""h: this page
s: save and leave
q: quit without saving
...1.: change the fourth label to 1
.-...: drop the second data 
(the two lines above can be merged as .-.1.)
empty: default is no change
*** No changes are made until saved ***
""")
                continue
            elif c == 'q':
                break
            elif c == 's':
                self.df = new_df.loc[~(new_df.notes == 'd'),[0,1]]
                self.df = self.df.reset_index(drop=True)
                for new_i in range(self.num):
                    if new_df.loc[new_i,'notes'] == 'd':
                        fn = new_df.iloc[new_i,0]
                        os.remove(os.path.join(self.path, fn))
                old_num = self.num
                self.num = self.df.shape[0]
                print("Changed %s labels and dropped %s pictures:"
                      %(np.sum(new_df.notes == 'r'),
                        np.sum(new_df.notes == 'd')))
                print("Number of images: %s -> %s"%(old_num, self.num))
                self.df.to_csv(os.path.join(self.path, self.name+'.csv'), 
                               header=False, 
                               index=False)
                print("New %s.csv written to %s."
                      %(self.name, self.path))
                new_df.to_csv(os.path.join(self.path, self.name+'_patch.csv'), 
                               header=False, 
                               index=False)
                print("The patch %s.csv written to %s."
                      %(self.name+'_patch', self.path))
                break
            elif c == '': ### default action
                i += 1
                continue
            elif all((d in '.-0123456789') for d in c):
                changes.append((i,c))
                for j,d in enumerate(c):
                    real_k = start + i*each_row + j
                    if d == '.':
                        continue
                    elif d == '-':
                        new_df.iloc[real_k,1] = -1
                        new_df.loc[real_k,'notes'] = 'd'
                        continue
                    elif d.isdigit():
                        if new_df.iloc[real_k,1] != float(d):
                            new_df.loc[real_k,'notes'] = 'r' 
                            new_df.iloc[real_k,1] = float(d)
                        continue
                    else:
                        print("Something is wrong.  Try again.")
                        break
            else:
                print("Not valid input.  Press h for help.")
                continue
    
    def merge_to(self, target='nsysu-digits'):
        exist_files = os.listdir(target)
        tar_name = os.path.split(target)[-1]
        if tar_name+'.csv' not in exist_files:
            start = 0
        else:
            nsysu = raw_data(target)
            start = nsysu.num
        
        ### expand DataFrame
        mix = self.df.copy()
        mix = mix.sample(frac=1)
        mix['new'] = ["%08d.png"%i for i in range(start, start + self.num)]
        # mix.sort_values('new')
        
        ### Move files
        for i in range(self.num):
            src,dst = mix.loc[i,0], mix.loc[i,'new']
            if dst in exist_files:
                print("File exists: %s -> %s"%(src,dst))
                break
        else:
            print("Copying files...", end=" ")
            for i in range(self.num):
                src,dst = mix.loc[i,0], mix.loc[i,'new']
                shutil.copyfile(os.path.join(self.path,src), 
                                os.path.join(target,dst))
            print("Done")
        
        ### Create or merge csv
        mix = mix.loc[:,['new',1]].rename({'new':0}, axis=1)
        mix[2] = self.name
        if start == 0:
            print("Creating csv...", end=" ")
            mix.to_csv(os.path.join(target, tar_name+'.csv'),
                       header=False,
                       index=False)
        else:
            print("Merging csv...", end=" ")
            pd.concat([nsysu.df,mix], 
                      ignore_index=True).to_csv(os.path.join(target, tar_name+'.csv'), 
                                                header=False, 
                                                index=False)
        print("Done")
