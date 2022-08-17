import glob
import os
import shutil
import csv
import pandas as pd
from opensoundscape.audio import Audio

text_files_dir = '/datadrive/azure_blob/Downloads/txt_files'
new_text_files_dir = '/datadrive/azure_blob/Downloads/txt_files_reformatted'
os.makedirs(new_text_files_dir, exist_ok=True)

# get list of full paths to wav files:
audio_files_dir = '/datadrive/azure_blob/Downloads/*/'
audio_files = glob.glob(f"{audio_files_dir}/*.wav")+glob.glob(f"{audio_files_dir}/*.WAV")
audio_files_name_only = [x.split('/')[-1] for x in audio_files]
# audio_files = [x.replace('.wav', '.WAV') for x in audio_files]

# create a dictionary that maps filenames to full paths:
audio_filename_to_full_path = {}
for full_path in audio_files:
    filename = full_path.split('/')[-1]
    audio_filename_to_full_path[filename]= full_path

header_row = 'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tannotation\tlow_f\thigh_f\n' 

with open('processed_files2.csv', newline='') as csvfile:
    data = csv.reader(csvfile)
    i = 0
    for row in data:
        #print(row)
        file = row[0]
        
        # skip the header row:
        i = i + 1
        if i == 1:
            continue
        
        # get full path to audio file:
        if 'SWIFT' in file or 'swift' in file:  
            file = file.replace('.WAV', '.wav')
        if file not in audio_files_name_only:
            print('{} in csv but not in disk; skipping'.format(file))
            continue
        audio_file_path = audio_filename_to_full_path[file]

        # get text file path:
        file_idx = file.replace('.wav', '').replace('.WAV', '')
        text_file_load_path = text_files_dir + '/' + file_idx + '.selections.txt'
        text_file_save_path = new_text_files_dir + '/' + file_idx + '.selections.txt'

        if row[3] == 'y':
            with open(text_file_load_path, 'r') as f:
                lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].rstrip()
                if i == 0:
                    lines[i] = lines[i] + '\tannotation\n'
                else:
                    lines[i] = lines[i] + '\t1\n'
            with open(text_file_save_path, 'w') as f:
                f.writelines(lines)

            #append column
            pass # dummy line that does nothing, just to keep the if statement happy
        elif row[3] == 'n':
            audio = Audio.from_file(audio_file_path)
            length = audio.duration()
            data_row = '1\t1\t1\t0\t'+str(length)+'\t0\t0\t24000'
            with open(text_file_save_path, 'w') as f:
                f.writelines([header_row, data_row])
    
    # name = file.split('/')[-1]
    # #instead of a single lookup value, map from the name to a dictionary containing the
    # # audio file and the text file
    # file_idx = name.replace('.wav', '').replace('.WAV', '')
    # # check if text file exists text_files_dir + '/' + name + '.selections.txt'
    # text_file_path = text_files_dir + '/' + file_idx + '.selections.txt'
    # file_exists = os.path.exists(text_file_path)
    # if file_exists:
    #   all_files[file_idx] = {'audio_file': file, 'text_file': text_file_path}
    # else:
    #     all_files[file_idx] = {'audio_file': file, 'text_file': ''}
    #     audio_file = Audio.from_file(file)
    #     length = audio.duration()
    #     header_row = 'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tannotation\n' 
    #     data_row = '1\t1\t1\t0\tlength\t0'
    #     # f = open(text_file_path, "w") 
    #     # f.writelines([header_row, data_row])
    #     # f.close()
    #     print(text_file_path)
    #     assert 0

