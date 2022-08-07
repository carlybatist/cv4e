from koogu.data import preprocess, feeder
from koogu.model import architectures
from koogu import train, assessments, recognize
from glob import glob
from matplotlib import pyplot as plt

raven_files_dir = '/Users/carlybatist/Documents/CV4Ecology/Practice'
audio_files_dir = '/Users/carlybatist/Documents/CV4Ecology/Practice'

annots_root = glob(f"{raven_files_dir}/*.txt")
audio_root = glob(f"{audio_files_dir}/*.wav")+glob(f"{audio_files_dir}/*.WAV")
annots_root

# Map audio files (or containing folders) to respective annotation files
audio_annot_list = [
    ['audiomoth1_20211212_050300', 'audiomoth1_20211212_050300.selections.txt'],
    ['audiomoth12_20220226_140500', 'audiomoth12_20220226_140500.selections.txt'],
    ['audiomoth2_20211217_061100', 'audiomoth2_20211217_061100.selections.txt'],
]

data_settings = {
    # Settings for handling raw audio
    'audio_settings': {
        'clip_length': 4.0,
        'clip_advance': 0.1,
        'desired_fs': 48000
    },

    # Settings for converting audio to a time-frequency representation
    'spec_settings': {
        'win_len': 0.128,
        'win_overlap_prc': 0.50,
    }
}

prepared_audio_dir = '~/prepared_data'

# Convert audio files into prepared data
clip_counts = preprocess.from_selection_table_map(
    data_settings['audio_settings'],
    audio_annot_list,
    audio_root, annots_root,
    output_root=prepared_audio_dir,
    negative_class_label='Other')