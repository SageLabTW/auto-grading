import os
import shutil
import pdf2image as p2i
from PIL import Image, ImageDraw, ImageFont
import pyzbar.pyzbar as zbar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### for email
import getpass
import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication

def clean(folder):
    """rm -rf folder"""
    shutil.rmtree(folder)

def get_server():
    print("Please enter your G-mail login information.")
    print("Visit https://support.google.com/mail/answer/185833?hl=en-GB to see how to set up app password.")
    user = input(prompt="username: ")
    pwd = getpass.getpass(prompt="app password: ") 
    try:
        server = smtplib.SMTP(host="smtp.gmail.com", port="587") # setup server
        server.ehlo()  # verify the connection
        server.starttls()  # use TLS
        server.login(user, pwd)  # login with credentials
        return server
    except Exception as e:
        print("Error message: ", e)

def quiz_result_content(to, pic_path):
    """
    Input:
        to: receiver's email address
        pic_path: the path to the attached picture
    """
    fn = os.path.split(pic_path)[-1]
    
    content = MIMEMultipart()  
    content["subject"] = "Quiz result" 
    content["from"] = "jephianlin@g-mail.nsysu.edu.tw" 
    content["to"] = to 
    content.attach(MIMEText("Contact jephianlin@gmail.com if you have any question.")) # content
    content.attach(MIMEImage(Path(pic_path).read_bytes(), Name=fn))
    # pictures attached by the next line cannot be opened directly on Gmail page
    # it can be downloaded though
    # content.attach(MIMEApplication(Path("sampleJ_grade_000.png").read_bytes(), Name='sample'))
    return content
            
def send_email(server, content):
    server.send_message(content)
    print("Sent!")

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
        
    def get_email(self, email_path):
        """create/update the self.full.email from email_path
        
        email_path is a csv file containing two columns (std_id, email_address)
        """
        try:
            emails = pd.read_csv(email_path, index_col=0, squeeze=True, header=None)    ### load key
        except FileNotFoundError:
            print('File not found: {}'.format(email_path))
            return 
        
        self.full['email'] = ['']*self.num
        for i in range(self.num):
            try:
                self.full.loc[i,'email'] = emails[self.full.loc[i, 'id']]
            except KeyError:
                self.full.loc[i,'email'] = -1
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
        
        self.full['graded_filename'] = ['']*self.num

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
            self.full.loc[i,'graded_filename'] = output_path.format(i)
        self.update_full()

    def group_email(self, receiver='all', mode='dry', test_receiver='jephianlin@gmail.com'):
        """
        receiver can be 'all' or 'v'
        mode can be 'dry', 'test', or 'send'
        use 'send' with caution
        """
        ### warning message
        if mode == 'send':
            print("You are going to send emails.  Sure? [y/N]")
            ans = input()
            if ans in ['y', 'Y']:
                server = get_server()
                print('Server ready.')
            elif ans in ['n', 'N', '', None]:
                return
            
        if mode == 'test':
            server = get_server()
            print('Server ready.')
        
        if receiver == 'v':
            mailing_list = self.full.loc[self.full['receiver'] == 'v']
        if receiver == 'all':
            mailing_list = self.full.copy()
            
        counter = 0
        for i in range(self.num):
            row = mailing_list.iloc[i]
            std_id = row['id']
            email = row['email']
            pic_path = row['graded_filename']
            
            if mode == 'test' and i == 0:
                content = quiz_result_content(test_receiver, pic_path)
                print('testing the first email to {}'.format(test_receiver))
                print('it was suppose go to {}<{}>'.format(std_id, email))
                send_email(server, content)
            
            content = quiz_result_content(email, pic_path)
            print("Sending email to {}<{}> attaching {}...".format(std_id, email, pic_path))
            if mode == 'send':
                send_email(server, content)
                counter += 1
            else:
                print('dry run')

        print("{} emails sent.".format(counter))
        
        if mode == 'send' or mode == 'test':
            server.close()
            
            
def get_record(base, first, *arg, to_csv=True):
    """
    Input:
        base: usually path to email.csv
        first: proj_name of the mandatory exam
        arg*: makeup exams
    Output:
        a DataFrame recording id, points, filename of students
    """
    base_df = pd.read_csv(base, header=None)
    first_df = raw_paper(first).full[['id','points','filename']]
    miss = [std_id for std_id in base_df[0].values if std_id not in first_df.id.values]
    ### miss_df has three columns ['id','points','filename']
    miss_df = pd.DataFrame({'id': miss})
    miss_df['points'] = 0
    miss_df['filename'] = 'miss'
    
    dfs = [raw_paper(proj_name).full[['id','points','filename']] for proj_name in arg]
    
    record = pd.concat([first_df, miss_df] + dfs, ignore_index=True)
    
    if to_csv:
        record.to_csv(first+'_record.csv')
    
    return record

def get_scores(df, proj_name=None, to_csv=True):
    """
    Input:
        df: a DataFrame generated by get_record
    """
    
    scores = df[['id','points']].groupby('id').mean().reset_index()
    
    if to_csv:
        scores.to_csv(proj_name+'_scores.csv')
    
    return scores
    
    
    