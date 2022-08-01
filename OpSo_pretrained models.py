from opensoundscape.torch.models.cnn import load_model
import opensoundscape
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import subprocess
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[15,5] #for large visuals
from opensoundscape.torch.models.cnn import CNN
from glob import glob
from opensoundscape.preprocess.utils import show_tensor_grid
from opensoundscape.torch.datasets import AudioSplittingDataset

# import & load model
CNN('resnet18',['classA','classB'],5.0).save('./temp.model')
model = load_model('./temp.model')

# use glob to create a list of all files matching a pattern in a folder:
audio_files = glob('./*.WAV') #match all .wav files in the current directory
audio_files

# generate predictions with the model
scores, _, _ = model.predict(audio_files)
scores.head()

# overlapping prediction clips
#scores, _, _ = model.predict(audio_files, overlap_fraction=0.5)
#scores.head()

# generate a dataset with the samples we wish to generate and the model's preprocessor
inspection_dataset = AudioSplittingDataset(audio_files, model.preprocessor)
inspection_dataset.bypass_augmentations = True

samples = [sample['X'] for sample in inspection_dataset]
_ = show_tensor_grid(samples,4)

# add a softmax layer to make the prediction scores for both classes sum to 1
# use the binary_preds argument to generate 0/1 predictions for each sample and class
# For presence/absence models, use the option binary_preds='single_target'
# For multi-class models, think about whether each clip should be labeled with 
# only one class (single target) or whether each clip could contain multiple classes (binary_preds='multi_target')

scores, binary_predictions, _ = model.predict(
    audio_files,
    activation_layer='softmax',
    binary_preds='single_target'
)
scores.head()
binary_predictions.head()
_ = plt.hist(scores['classA'],bins=20)
_ = plt.xlabel('softmax score for classA')