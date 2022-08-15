import glob
import os
import shutil
import csv
import pandas as pd

#Set directory to everything within Downloads folder
audio_files_dir = '/datadrive/azure_blob/Downloads/*/'
#Make list of all wav files in all folders within Downloads
audio_files = glob.glob(f"{audio_files_dir}/*.wav")+glob.glob(f"{audio_files_dir}/*.WAV")
#Make a list of all the txt files
text_files_dir = '/datadrive/azure_blob/Downloads/txt_files'
raven_files = glob.glob(f"{text_files_dir}/*.selections.txt")

#Make dictionary 
# all_files = {}

# for file in audio_files:
#     name = file.split('/')[-1]
#     #instead of a single lookup value, map from the name to a dictionary containing the
#     # audio file and the text file
#     file_idx = name.replace('.wav', '').replace('.WAV', '')
#     # check if text file exists text_files_dir + '/' + name + '.selections.txt'
#     text_file_path = text_files_dir + '/' + file_idx + '.selections.txt'
#     file_exists = os.path.exists(text_file_path)
#     if file_exists:
#       all_files[file_idx] = {'audio_file': file, 'text_file': text_file_path}
#     else:
#         all_files[file_idx] = {'audio_file': file, 'text_file': ''}
#         audio_file = Audio.from_file(file)
#         length = audio.duration()
#         header_row = 'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tannotation\n' 
#         data_row = '1\t1\t1\t0\tlength\t0'
#         # f = open(text_file_path, "w") 
#         # f.writelines([header_row, data_row])
#         # f.close()
#         print(text_file_path)
#         assert 0

# empty = '/datadrive/azure_blob/Downloads/empty_1.txt'

# for file_idx in all_files:
#     if all_files[file_idx]['text_file'] == '':
#         new_file_path = text_files_dir + '/' + file_idx + '.selections.txt'
#         print(new_file_path)
#         shutil.copy(empty, new_file_path)
#         #print(x, all_files[x])

train = []
val = []
test = []

with open('processed_files2.csv', newline='') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        file_idx = row[0].replace('.wav', '').replace('.WAV', '')
        day=row[2]
        #print(file_idx, type(day))
        if day == '1':
            train.append(file_idx)
        elif day == '2':
            val.append(file_idx)
        else:
            test.append(file_idx)

print(len(train), len(val), len(test))

#make an array (list of lists) from our all_files dictionary 
# each "row" should be [file_idx, audio_file, text file]
df_list = []
for file_idx in all_files:
    df_list.append([file_idx, all_files[file_idx]['audio_file'], all_files[file_idx]['text_file']])

print(len(df_list))