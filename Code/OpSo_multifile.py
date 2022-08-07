from turtle import colormode
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.annotations import BoxedAnnotations
from opensoundscape.helpers import generate_clip_times_df
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

# set directory to find .wav and .txt files
raven_files_dir = '/Users/carlybatist/Documents/CV4Ecology/Practice'
audio_files_dir = '/Users/carlybatist/Documents/CV4Ecology/Practice'

# find all audio/txt files files
raven_files = glob(f"{raven_files_dir}/*.selections.txt")
print(f"found {len(raven_files)} annotation files")
audio_files = glob(f"{audio_files_dir}/*.wav")+glob(f"{audio_files_dir}/*.WAV")
print(f"found {len(audio_files)} audio files")
print(raven_files)
print(audio_files)
# specify a list of classes to include in a subset
#classes_to_keep = ['?']
#annotations_only_unsure = annotations.subset(classes_to_keep)
#annotations_only_unsure.df

# pair up the raven and audio files based on the audio file name
audio_df = pd.DataFrame({'audio_file':audio_files})
print(len(audio_df))
audio_df
audio_df.index = [Path(f).stem for f in audio_files]
raven_df = pd.DataFrame({'raven_file':raven_files})
raven_df
raven_df.index = [Path(f).stem.split('.Table')[0].replace('.selections','') for f in raven_files]
paired_df = audio_df.join(raven_df,how='outer')
paired_df

#check that there aren't duplicate annotation file names
print('\n raven files with duplicate names:')
raven_df[raven_df.index.duplicated(keep=False)]
#check that there aren't duplicate audio file names
print('\n audio files with duplicate names:')
audio_df[audio_df.index.duplicated(keep=False)]

#choose settings for audio splitting
clip_duration = 5
clip_overlap = 0
final_clip = None
clip_dir = Path('./temp_clips')
clip_dir.mkdir(exist_ok=True)

#choose settings for annotation splitting
classes = None#['GWWA_song','GWWA_dzit'] #list of all classes, or None
min_label_overlap = 0.1

#store the label dataframes from each audio file so that we can aggregate them later
#Note: if you have a huge number (millions) of annotations, this might get very large.
#an alternative would be to save the individual dataframes to files, then concatenate them later.
all_labels = []

cnt = 0
#prefix=pd.DataFrame({'audio_file':audio_files})

for i, row in paired_df.iterrows():
    #load the audio into an Audio object
    audio = Audio.from_file(row['audio_file'])

    #in this example, only the first 60 seconds of audio is annotated
    #so trim the audio to 60 seconds max
    audio = audio.trim(0,60)

    #split the audio and save the clips
    clip_df = audio.split_and_save(
        clip_dir,
        prefix=row.name,
        clip_duration=clip_duration,
        clip_overlap=clip_overlap,
        final_clip=final_clip,
        dry_run=False
    )

    #load the annotation file into a BoxedAnnotation object
    annotations = BoxedAnnotations.from_raven_file(row['raven_file'],annotation_column='Selection')

    #since we trimmed the audio, we'll also trim the annotations for consistency
    annotations = annotations.trim(0,60)

    #split the annotations to match the audio
    #we choose to keep_index=True so that we retain the audio clip's path in the final label dataframe
    labels = annotations.one_hot_labels_like(clip_df,classes=classes,min_label_overlap=min_label_overlap,keep_index=True)
    labels['lemur'] = labels.sum(axis=1)
    labels = labels['lemur']

    #since we have saved short audio clips, we can discard the start_time and end_time indices
    labels = labels.reset_index(level=[1,2],drop=True)
    all_labels.append(labels)

    cnt+=1
    if cnt>2:
        break

#make one big dataframe with all of the labels. We could use this for training, for instance.
all_labels = pd.concat(all_labels)
print(all_labels)
all_labels.to_csv("practice_one_hot_encoded_labels.csv")

# one hot encoded labels
# file,cardinal,jay
# file1.wav,1,0
# file2.wav,0,0

#sanity check
# plot spectrograms for 3 random positive clips
positives = all_labels[all_labels==1].sample(3,random_state=0)
print("spectrograms of 3 random positive clips (label=1)")
for positive_clip in positives.index.values:
    print(positive_clip)
    spec = Spectrogram.from_audio(Audio.from_file(positive_clip))
    spec.bandpass(0, 8000).plot(colormode)

# plot spectrograms for 3 random negative clips
negatives = all_labels[all_labels==0].sample(3,random_state=0)
print("spectrogram of 3 random negative clips (label=0)")
for negative_clip in negatives.index.values:
    Spectrogram.from_audio(Audio.from_file(negative_clip)).plot()

