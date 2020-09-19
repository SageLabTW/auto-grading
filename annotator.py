import os
import shutil
import pdf2image as p2i
from PIL import Image, ImageDraw, ImageFont
import pyzbar.pyzbar as zbar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clean(folder):
    """rm -rf folder"""
    shutil.rmtree(folder)



class raw_paper:
    def __init__(self, path):
        """initiate the object with path = folder_name"""
        self.path = path
        self.name = os.path.split(path)[-1]
        self.path_full = os.path.join(path, self.name+'_full.csv')
        self.full = pd.read_csv(self.path_full) 
#         self.label = pd.read_csv(os.path.join(path, self.name+'_pred.csv'), 
#                               header=None)
#         self.answer = pd.read_csv(os.path.join(path, self.name+'_qr.csv'), 
#                               header=None)
        self.num = self.full.shape[0]
    
    def update_full(self):
        self.full.to_csv(self.path_full, index=False)
        
    def get_label(self):
        """import label from {path}.csv
        
        Usually, one have to do 
            raw = ex.raw_data(path)
            raw.labeler()
        to update the label.
        """
        raw_df = pd.read_csv(os.path.join(self.path, self.name+'.csv'), header=None) 
        if raw_df.shape[0] == self.num:
            self.full['std_ans'] = raw_df[1]
            self.update_full()
        else:
            print('{} and {} has different number of rows.'.format(
                os.path.join(self.path, self.name+'.csv'), 
                self.path_full))
            print('Check and try again.')

    def get_cor_ans(self, key_path='default'):
        """update self.full.cor_ans by self.full.qr"""
        if key_path == 'default':
            key_path = os.path.join('keys','{}_key.csv'.format(self.name))
        try:
            key = pd.read_csv(key_path, index_col=0, squeeze=True)    ### load key
        except FileNotFoundError:
            print('File not found: {}'.format(key_path))
            return 
        
        for i in range(self.num):
            try:
                self.full.loc[i,'cor_ans'] = key[self.full.loc[i, 'qr']]
            except KeyError:
                self.full.loc[i,'cor_ans'] = -1
        self.update_full()
        
    def grade(self):
        """update self.full.points"""
        self.full.points = 5 * (self.full.std_ans == self.full.cor_ans)
        self.update_full()
                  
    def annotate(self, output_folder = 'default', filename = 'default'):
        # 建立資料夾，預設為 self.name + grade
        if output_folder == 'default':
            fn = os.path.split(self.path)[-1]
            output_folder = os.path.splitext(fn)[0] + '_grade'
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
        else:
            filename = filename + '_{:03d}.png'
        output_path = os.path.join(output_folder, filename)

        font = ImageFont.truetype('font_style/Montserrat-Regular.ttf', size = 50)

        for i in range(self.num):
            row = self.full.iloc[i]
            fn = row['filename']
            im = Image.open(os.path.join(self.path, fn))
            draw = ImageDraw.Draw(im)
            num = int(row['std_ans'])
            ans = int(row['cor_ans'])
            point = int(row['points'])
            std_id = row['id']
            draw.text((200,100), "point = {}".format(point), font=font, fill='red')
            draw.text((1300,100), "{}".format(std_id), font=font, fill='red')
            draw.text((400,2200), "answer = {}".format(ans), font=font, fill='red')
            draw.text((1000,2200), "digits = {}".format(num), font=font, fill='red')
            im.save(output_path.format(i))
