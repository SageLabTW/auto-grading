
import os
import shutil
import pdf2image as p2i

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean(folder):
    """rm -rf folder"""
    shutil.rmtree(folder)

def extract(path, box=(1377,2047,1503,2173), get_key=False, output_folder='default', filename='default', pages=None):
    """Write ppm files and a csv file to output_folder.
    
    Input:
        path: path of pdf file
        output_folder: name of output folder, default name is same as pdf file
        filename: suffix of output filename, default name is {output_folder}_{}.pgm
        box: (left, top, right, bottom) for cropping checkcode
            108 fake: ???
            108~109 real: (1377,2047,1503,2173)
            a4: (1417.78, 2102.78, 1575.26, 2260.26) not tested
    Note: 
        a4 page size = 29.7 x 21 cm; 11.69 x 8.27 in; 2339 x 1654 in 200 dpi
        1cm = 0.3937 inches = 78.74 pixels in 200 dpi
    Output:
        None
    """
    
    
    # 建立資料夾，預設為 pdf 檔名
    if output_folder == 'default':
        fn = os.path.split(path)[-1] ### Jep: use os to analyze the path
        output_folder = os.path.splitext(fn)[0]
    try:
        os.mkdir(output_folder)
    except FileExistsError: ### make exception precise
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
        filename = output_folder + '_{:03d}.png' ### Jep: change {} to {:03d}
    else:
        filename = filename + '_{:03d}.png'
    output_path = os.path.join(output_folder, filename)
    
    # 將 pdf 逐頁拆分並轉換成 jpg 並儲存
    if pages != None:
        imgs = p2i.convert_from_path(path, grayscale = True, 
                                     first_page=pages[0], last_page=pages[1]) ### Jep: use default pgm format
    else:
        imgs = p2i.convert_from_path(path, grayscale = True) ### Jep: use default pgm format
    
    ### create csv
    num_imgs = len(imgs)
    df = pd.DataFrame([
            [filename.format(i) for i in range(num_imgs)], 
            [None]*num_imgs
        ])
    df.T.to_csv(os.path.join(output_folder, output_folder+'.csv'), 
                header=False, 
                index=False)
    
    for i, im in enumerate(imgs):
        ### extract checkcode
        checkcode = im.crop(box)
        checkcode = checkcode.resize((28,28))
        checkcode.save(output_path.format(i))
        
        ### extract key
        ### TBD
        
class raw_data:
    def __init__(self, path):
        """initiate the object with path = folder_name"""
        self.path = path
        self.name = os.path.split(path)[-1]
        self.df = pd.read_csv(os.path.join(path, self.name+'.csv'), 
                              header=None)
        self.num = self.df.shape[0]
    
    def examine(self, start=None, end=None, each_row=5, size=2):
        if start == None:
            start = 0
        if end == None:
            end = self.num

        total = end - start
        rows = total // each_row
        if total % each_row != 0:
            rows += 1
            
        fig,axs = plt.subplots(rows, each_row, 
                               figsize=(size*each_row,size*rows))

        for k in range(total):
            i,j = k//each_row, k%each_row
            fn = self.df.iloc[k,0]
            ax = axs[i][j]
            ax.axis('off')
            img = plt.imread(os.path.join(self.path, fn))
            ax.imshow(img, cmap='Greys_r', vmin=0, vmax=1)
            ax.set_title('%s'%self.df.iloc[k,1])
            ax.text(0, 27, fn)
        
        return fig
    
    def labeler(self, start=None, end=None, each_row=5, size=2):
        #TBD
        pass
    
    def merge_to(self, target='nsysu_digits'):
        #shuffle and keep label only
        #TBD
        pass