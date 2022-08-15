from koogu.data import preprocess, feeder
from koogu.model import architectures
from koogu import train, assessments, recognize
from matplotlib import pyplot as plt   

# The root directories under which the training data (audio files and
# corresponding annotation files) are available.
audio_root = '/home/shyam/projects/NARW/data/train_audio'
annots_root = '/home/shyam/projects/NARW/data/train_annotations'

# Map audio files (or containing folders) to respective annotation files
audio_annot_list = [
    ['NOPP6_EST_20090328', 'NOPP6_20090328_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090329', 'NOPP6_20090329_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090330', 'NOPP6_20090330_RW_upcalls.selections.txt'],
    ['NOPP6_EST_20090331', 'NOPP6_20090331_RW_upcalls.selections.txt'],
]

data_settings = {
    # Settings for handling raw audio
    'audio_settings': {
        'clip_length': 2.0,
        'clip_advance': 0.4,
        'desired_fs': 1000
    },

    # Settings for converting audio to a time-frequency representation
    'spec_settings': {
        'win_len': 0.128,
        'win_overlap_prc': 0.75,
        'bandwidth_clip': [46, 391]
    }
}


#preprocessing step will split up the audio files into clips (defined by data_settings['audio_settings']), 
#match available annotations to the clips, and 
#mark each clip to indicate if it matched one or more annotations

# Path to the directory where pre-processed data will be written.
# Directory will be created if it doesn't exist.
prepared_audio_dir = '/home/shyam/projects/NARW/prepared_data'

# Convert audio files into prepared data
#we can consider all un-annotated time periods in the recordings as inputs 
#for the negative class (by setting the parameter negative_class_label
clip_counts = preprocess.from_selection_table_map(
    data_settings['audio_settings'],
    audio_annot_list,
    audio_root, annots_root,
    output_root=prepared_audio_dir,
    negative_class_label='Other')

# Display counts of how many inputs we got per class
for label, count in clip_counts.items():
    print(f'{label:<10s}: {count:d}')


#we define a feeder that efficiently feeds all the pre-processed clips, in batches, 
#to the training/validation pipeline.
#feeder also transforms the audio clips into spectrograms.
data_feeder = feeder.SpectralDataFeeder(
    prepared_audio_dir,                        # where the prepared clips are at
    data_settings['audio_settings']['desired_fs'],
    data_settings['spec_settings'],
    validation_split=0.2,                     # set aside 20% for validation
    max_clips_per_class=10000                  # use up to 10k inputs per class
)