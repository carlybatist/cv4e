from glob import glob
text_files_dir = '/datadrive/azure_blob/Downloads/'
raven_files = glob(f"{text_files_dir}/*.selections.txt")
print(raven_files)

import os
import shutil

os.mkdir(text_files_dir+'txt_files')

for txt in raven_files:
    name = txt.split('/')[-1].replace('txt_files','')
    new_path = text_files_dir + 'txt_files/' + name
    shutil.move(txt, new_path)
