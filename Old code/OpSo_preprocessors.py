import pwd
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.torch.datasets import AudioFileDataset, AudioSplittingDataset
from opensoundscape.preprocess.utils import show_tensor, show_tensor_grid
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import random
import subprocess
from matplotlib import pyplot as plt
from opensoundscape.preprocess.utils import show_tensor_grid
plt.rcParams['figure.figsize']=[15,5] #for large visuals

# load dataframes
#load one-hot labels dataframe
labels = pd.read_csv('practice_one_hot_encoded_labels.csv').set_index('file')

# prepend the folder location to the file paths
labels.index = pd.Series(labels.index).apply(lambda f: './'+f)
labels.head()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

pre = SpectrogramPreprocessor(sample_duration=3.0)
dataset = AudioFileDataset(labels,pre)
dataset[0] #loads and preprocesses the sample at row 0 of dataset.df

# visualize multiple samples 
pre = SpectrogramPreprocessor(sample_duration=3.0)
dataset = AudioFileDataset(labels,pre)
tensors = [dataset[i]['X'] for i in range(9)]
sample_labels = [dataset[i]['y'] for i in range(9)]
_ = show_tensor_grid(tensors,3,labels=sample_labels)

# without augmentation
dataset.bypass_augmentations = True
tensors = [dataset[i]['X'] for i in range(9)]
sample_labels = [dataset[i]['y'] for i in range(9)]
_ = show_tensor_grid(tensors,3,labels=sample_labels)

# subset samples from a Dataset

len(dataset)
len(dataset.head(10))
len(dataset.sample(n=10))
len(dataset.sample(frac=0.5))

# loading many fixed-duration samples from longer audio files
pwd
prediction_df = pd.DataFrame(index=['./audiomoth12_20220226_140500.wav'])
prediction_df
pre = SpectrogramPreprocessor(sample_duration=3.0)
splitting_dataset = AudioSplittingDataset(prediction_df,pre,overlap_fraction=0.1)
splitting_dataset.bypass_augmentations = True
#get the first 9 samples and plot them
tensors = [splitting_dataset[i]['X'] for i in range(9)]
_ = show_tensor_grid(tensors,3)

# preprocessor pipelines

preprocessor = SpectrogramPreprocessor(sample_duration=3)
preprocessor.pipeline
# view default parameters for an Action
preprocessor.pipeline.to_spec.params
# example of modifying parameters with the Action’s .set() method:
preprocessor.pipeline.to_spec.set(dB_scale=False)
# example of modifying parameters by accessing parameter directly (params is pd Series)
preprocessor.pipeline.to_spec.params.window_samples = 512
preprocessor.pipeline.to_spec.params['overlap_fraction'] = 0.75
preprocessor.pipeline.to_spec.params

#bypass actions
preprocessor = SpectrogramPreprocessor(sample_duration=3.0)
#turn off augmentations other than noise
preprocessor.pipeline.add_noise.bypass=True
preprocessor.pipeline.time_mask.bypass=True
preprocessor.pipeline.frequency_mask.bypass=True
#printing the pipeline will show which actions are bypassed
preprocessor.pipeline

# create a Dataset with this preprocessor & our label dataframe
dataset = AudioFileDataset(labels,preprocessor)
print('random affine off')
preprocessor.pipeline.random_affine.bypass = True
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)
plt.show()
print('random affine on')
preprocessor.pipeline.random_affine.bypass = False
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)

#view whether actions are on or off (examples of)
preprocessor.pipeline.load_audio.bypass
preprocessor.pipeline.frequency_mask.bypass

# modifying the pipeline

#example: return Spectrogram instead of Tensor
#initialize a preprocessor
preprocessor = SpectrogramPreprocessor(2.0)
print('original pipeline:')
[print(p) for p in pre.pipeline]

#overwrite the pipeline with a slice of the original pipeline
print('\nnew pipeline:')
preprocessor.pipeline = preprocessor.pipeline[0:4]
[print(p) for p in preprocessor.pipeline]

print('\nWe now have a preprocessor that returns Spectrograms instead of Tensors:')
dataset = AudioFileDataset(labels,preprocessor)
print(f"Type of returned sample: {type(dataset[0]['X'])}")
dataset[0]['X'].plot()

# modifying actions

#resample all loaded audio to a specified rate during the load_audio action
pre = SpectrogramPreprocessor(sample_duration=3)
pre.pipeline.load_audio.set(sample_rate=24000)

#modify spectrogram window length and overlap
dataset = AudioFileDataset(labels,SpectrogramPreprocessor(sample_duration=3))
dataset.bypass_augmentations=True
print('default parameters:')
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)
plt.show()
print('high time resolution, low frequency resolution:')
dataset.preprocessor.pipeline.to_spec.set(window_samples=64)
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)

#bandpass filters
dataset = AudioFileDataset(labels, SpectrogramPreprocessor(3.0))
print('default parameters:')
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)
print('bandpassed to 2-4 kHz:')
dataset.preprocessor.pipeline.bandpass.set(min_f=2000,max_f=4000)
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)

#turn all augmentation off
dataset = AudioFileDataset(labels, SpectrogramPreprocessor(3.0))
dataset.bypass_augmentations = True
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)
#or on
dataset.bypass_augmentations = False
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)

# modify augmentation parameters

#initialize a preprocessor
preprocessor = SpectrogramPreprocessor(3.0)
#turn off augmentations other than overlay
preprocessor.pipeline.random_affine.bypass=True
preprocessor.pipeline.time_mask.bypass=True
preprocessor.pipeline.add_noise.bypass=True
# allow up to 20 horizontal masks, each spanning up to 0.1x the height of the image.
preprocessor.pipeline.frequency_mask.set(max_width = 0.03, max_masks=20)
#preprocess the same sample 4 times
dataset = AudioFileDataset(labels,preprocessor)
tensors = [dataset[0]['X'] for i in range(4)]
fig = show_tensor_grid(tensors,2)
plt.show()

# turn off frequency mask and turn on gaussian noise
dataset.preprocessor.pipeline.add_noise.bypass = False
dataset.preprocessor.pipeline.frequency_mask.bypass =True
# increase the intensity of gaussian noise added to the image
dataset.preprocessor.pipeline.add_noise.set(std=0.2)
show_tensor(dataset[0]['X'],invert=True,transform_from_zero_centered=True)
