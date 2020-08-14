
import os
import shutil
import pdf2image as p2i

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


def find_box(img, threshold = 180):
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
    box = (c-130, r-130, c-10, r-10)
    return box


    
def extract(path, box=(1377,2047,1503,2173), get_key=False, output_folder='default', filename='default', pages=None):
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
    else:
        filename = filename + '_{:03d}.png'
    output_path = os.path.join(output_folder, filename)
    
    # 將 pdf 逐頁拆分並轉換成 jpg 並儲存
    if pages != None:
        imgs = p2i.convert_from_path(path, grayscale = True, 
                                     first_page=pages[0], last_page=pages[1])
    else:
        imgs = p2i.convert_from_path(path, grayscale = True)
    
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
        ### find box
        box = find_box(im)
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
        
        while review:
            if i >= rows:
                print('Reaching the end.  [s]ave or [q]uit?')
            else:
                self.examine(start + i*each_row, 
                             min(start + (i+1)*each_row, total), 
                             each_row, size, 
                             label=new_df)

                print('Give me five digits for your changes: [h] help')

            c = input()
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
            elif len(c) == each_row or i == rows-1:
                changes.append(c)
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
        mix.sort_values('new')
        
        ### Move files
        for i in range(self.num):
            dst = mix.loc[i,'new']
            if dst in exist_files:
                print("File exists: %s -> %s"%(src,dst))
                break
        else:
            print("Copying files...", end=" ")
            for i in range(self.num):
                src,dst = mix.iloc[i,0], mix.loc[i,'new']
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
        