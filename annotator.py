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
        self.label = pd.read_csv(os.path.join(path, self.name+'_label.csv'), 
                              header=None)
        self.answer = pd.read_csv(os.path.join(path, self.name+'_qr.csv'), 
                              header=None)
        self.num = self.answer.shape[0]
        
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
            fn = 'paper_' + self.label.iloc[i,0]
            im = Image.open(os.path.join(self.path, fn))
            draw = ImageDraw.Draw(im)
            num = self.label.iloc[i,1]
            ans = self.answer.iloc[i,1]
            if num == ans:
                point = 5
            else:
                point = 0
            draw.text((1300,2200), "digits = {}".format(num), font=font)
            draw.text((100,2200), "answer = {}".format(ans), font=font)
            draw.text((1300,100), "point = {}".format(point), font=font)
            im.save(output_path.format(i))
