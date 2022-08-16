import torch
import torchaudio
import numpy as np
import os

def chunk_audio(audio_tensor, window_samples, overlap_samples):
    # Skips final chunk if there are leftover samples. 
    chunks = []
    cur_init = 0
    cur_final = window_samples
    delta = int(window_samples - overlap_samples)
    starts = []
    stops = []
    while cur_final <= len(audio_tensor):
        cur_chunk = audio_tensor[cur_init:cur_final]
        chunks.append(cur_chunk)
        starts.append(cur_init)
        stops.append(cur_final)
        cur_init += delta
        cur_final += delta
    return chunks, starts, stops

def get_clip(audio_path, range_sec=None, out_fs=None):
    waveform, sample_rate = torchaudio.load(audio_path)
    num_channels, num_samples = waveform.size()
    assert num_channels == 1
    if range_sec is not None:
        start_sample = int(np.round(sample_rate * range_sec[0]))
        stop_sample = int(np.round(sample_rate * range_sec[1]))
    else:
        start_sample = 0
        stop_sample = num_samples - 1
    clip = waveform[0, start_sample:stop_sample]
    if out_fs is not None:
        resampler = torchaudio.transforms.Resample(sample_rate, out_fs, dtype=clip.dtype)
        clip = resampler(clip)
        sample_rate = out_fs
    return clip, sample_rate

# def parse_raven_label_file(label_file_path):
#     with open(label_file_path, 'r') as f:
#         lines = f.readlines()
#     lines = [l.rstrip() for l in lines]
#     lines = [l.split('\t') for l in lines]
#     assert l[0][3] == 'Begin Time (s)'
#     assert l[0][4] == 'End Time (s)'
#     assert l[0][7] == 'annotation'
#     lines = lines[1:] # get rid of header row
#     lines = [(float(l[3]), float(l[4]), int(l[7])) for l in lines]
#     return lines

# audio_file_list = [
#     'temp_data/audiomoth1_20211212_050200.WAV',
#     'temp_data/audiomoth1_20211212_050300.WAV'
# ]
# label_file_list = [
#     'temp_data/audiomoth1_20211212_050200.selections.txt',
#     'temp_data/audiomoth1_20211212_050300.selections.txt'
# ]

# csv with time stamps for positives
# csv with file-level yes/no 

load_base = 'temp_data'
save_dir = 'temp_data_save'

window_length_sec = 5.0
overlap_length_sec = 0.0
out_fs = 32000

os.makedirs(save_dir, exist_ok=True)

# load mapping from filename to path:
with open('temp_data/paired2.csv', 'r') as f:
    path_lines = f.readlines()
path_lines = path_lines[1:]
path_lines = [l.split(',') for l in path_lines]
path_map = {l[0]: l[1] for l in path_lines}

# generate list of all files:
with open('temp_data/processed_files2.csv', 'r') as f:
    all_files = f.readlines()
days = [l.split(',')[2] for l in all_files]
days = days[1:]
days = [int(l) for l in days]
all_files = [l.split(',')[0] for l in all_files]
all_files = all_files[1:]

# parse annotations of calls:
with open('temp_data/timestamps2.csv', 'r') as f:
    annotations = f.readlines()
annotations = [l.rstrip() for l in annotations]
annotations = annotations[1:]
annotations = [l.split(',') for l in annotations]
annotations = [(str(l[0]), float(l[4]), float(l[5])) for l in annotations]
annotations_dict = {l[0]: [] for l in annotations}
for l in annotations:
    annotations_dict[l[0]].append((l[1], l[2]))

# 1 train
# 2 val
# else test

out_filenames = []
out_labels = []
out_days = []

for k, audio_file_name in enumerate(all_files):
    
    audio_file_path = path_map[audio_file_name]
    
    info = torchaudio.info(os.path.join(load_base, audio_file_path))
    length_sec = info.num_frames / info.sample_rate
    
    try:
        pos_intervals = annotations_dict[audio_file_name]
    except:
        pos_intervals = []
        neg_intervals = [(0.0, length_sec)]
    
    if len(pos_intervals) > 0:
        start_times = [x[0] for x in pos_intervals]
        idx_srt = np.argsort(start_times)
        pos_intervals = [pos_intervals[i] for i in idx_srt]
        
        '''
        generate negative intervals where appropriate:
        '''
        neg_intervals = []
        
        if pos_intervals[0][0] > 0:
            neg_intervals.append((0.0, pos_intervals[0][0]))
        
        for i in range(len(pos_intervals)-1):
            neg_intervals.append(pos_intervals[i][1], pos_intervals[i+1][0])
        
        if pos_intervals[-1][1] < length_sec:
            neg_intervals.append((pos_intervals[-1][1], length_sec))
    
    '''
    check reasonableness:
    '''
    
    length_summed = 0.0
    for I in pos_intervals:
        length_summed += I[1] - I[0]
    for I in neg_intervals:
        length_summed += I[1] - I[0]
    if length_summed != length_sec:
        print('The intervals do not line up!')
        print(audio_file_name)
        print('total interval length: {}'.format(length_summed))
        print('file length: {}'.format(length_sec))
        print('positive intervals:')
        print(pos_intervals)
        print('negative intervals:')
        print(neg_intervals)
        assert 0
        
    '''
    save windows:
    '''
    
    for I in pos_intervals:
        clip, fs = get_clip(os.path.join(load_base, audio_file_path), I, out_fs)
        window_length_samples = int(np.round(window_length_sec * fs))
        overlap_length_samples = int(np.round(overlap_length_sec * fs))
        chunks, starts, stops = chunk_audio(clip, window_length_samples, overlap_length_samples)
        for i, cur_chunk in enumerate(chunks):
            cur_name, cur_ext = audio_file_name.split('.')
            save_name = cur_name + '_' + str(starts[i]) + '_' + str(stops[i]) + '.' + cur_ext
            save_path = os.path.join(save_dir, save_name)
            torchaudio.save(save_path, cur_chunk.unsqueeze(0), fs)
            out_filenames.append(save_name)
            out_labels.append(1)
            out_days.append(days[k])
    
    for I in neg_intervals:
        clip, fs = get_clip(os.path.join(load_base, audio_file_path), I, out_fs)
        window_length_samples = int(np.round(window_length_sec * fs))
        overlap_length_samples = int(np.round(overlap_length_sec * fs))
        chunks, starts, stops = chunk_audio(clip, window_length_samples, overlap_length_samples)
        for i, cur_chunk in enumerate(chunks):
            cur_name, cur_ext = audio_file_name.split('.')
            save_name = cur_name + '_' + str(starts[i]) + '_' + str(stops[i]) + '.' + cur_ext
            save_path = os.path.join(save_dir, save_name)
            torchaudio.save(save_path, cur_chunk.unsqueeze(0), fs)
            out_filenames.append(save_name)
            out_labels.append(0)
            out_days.append(days[k])
    
    # progress:
    print('progress: {:.1f}/100'.format(100 * k / len(all_files)))
    

# train / val / test splits:

out_lines = [x + ',' + str(y) + '\n' for x, y in zip(out_filenames, out_labels)]

with open('chunk_labels_train.txt', 'w') as f:
    f.writelines([out_lines[i] for i in range(len(out_lines)) if out_days[i] == 1])

with open('chunk_labels_val.txt', 'w') as f:
    f.writelines([out_lines[i] for i in range(len(out_lines)) if out_days[i] == 2])

with open('chunk_labels_test.txt', 'w') as f:
    f.writelines([out_lines[i] for i in range(len(out_lines)) if out_days[i] == 3 or out_days[i] == 4])
