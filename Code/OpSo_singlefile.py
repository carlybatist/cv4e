from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.annotations import BoxedAnnotations
from opensoundscape.helpers import generate_clip_times_df
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

# load audio file as Audio object
audio = Audio.from_file("audiomoth2_20211217_061100.WAV")

print(f"How many samples does this audio object have? {len(audio.samples)}")
print(f"What is the sampling rate? {audio.sample_rate}")

# get duration
length = audio.duration()
print(length)

# trim file
trimmed = audio.trim(0,5)
trimmed.duration()

# split audio
clips, clip_df = audio.split(clip_duration=5,clip_overlap=0,final_clip=None)
#check the duration of the Audio object in the first returned element
print(f"duration of first clip: {clips[0].duration()}")
print(f"head of clip_df")
clip_df.head(3)

# create Spectrogram object from Audio object
spectrogram_object = Spectrogram.from_audio(audio)
spectrogram_object.plot()
plt.show()

#Larger value for window_samples –> higher frequency resolution (more rows in a single spectrogram column)
#Smaller value for window_samples –> higher time resolution (more columns in the spectrogram per second)
spec = Spectrogram.from_audio(audio, window_samples=55, overlap_samples=0)
spec.plot()
plt.show()

# save spectrogram to file
spectrogram_image = spectrogram_object.to_image()
print("Type of `spectrogram_audio` (before conversion):", type(spectrogram_object))
print("Type of `spectrogram_image` (after conversion):", type(spectrogram_image))
image_shape = (512, 1028) #(height, length)
spectrogram_image = spectrogram_object.to_image(shape=image_shape)
spectrogram_image.save('./saved.png')

# resample
audio_object_resample = Audio.from_file(audio_filename, sample_rate=22050)
audio_object_resample.sample_rate

# generate frequency spectrum
fft_spectrum, frequencies = trimmed.spectrum()

# bandpass
low_freq = 5500
high_freq = 9500
spec_bandpassed = spectrogram_object.bandpass(low_freq, high_freq)
spec_bandpassed.plot()
plt.show()

# calculate amplitude
high_freq_amplitude = spectrogram_object.amplitude()
# plot
plt.plot(spectrogram_object.times,high_freq_amplitude)
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.show()

# create an object from Raven file
audio = Audio.from_file("audiomoth2_20211217_061100.WAV")
annotation_file = 'audiomoth2_20211217_061100.selections.txt'
annotations = BoxedAnnotations.from_raven_file(annotation_file, annotation_column='Selection')

# inspect the object's .df attribute, which contains the table of annotations
print(annotations.df.head())

# splitting annotations along with audio, to train models on short segments
# split the audio into 5 second clips with no overlap (we use _ because we don't really need to save the audio clip objects for this demo
_, clip_df = audio.split(clip_duration=5, clip_overlap=0)
labels_df = annotations.one_hot_labels_like(clip_df,min_label_overlap=0.25, classes=None)
audio.split
#the returned dataframe of one-hot labels (0/1 for each class and each clip) has rows corresponding to each audio clip
labels_df.head()

# label_cols = labels_df.columns
# Add a column named lemur containing the annotation (1/0) for all selections
labels_df['lemur'] = labels_df.sum(axis=1)
labels_df = labels_df['lemur']
labels_df
