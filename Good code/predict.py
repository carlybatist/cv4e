'''
    Predict script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.
    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

# let's import our own classes and functions!
from util import init_seed
from dataset import AudioDataset
from model import CarNet


def create_dataloader(cfg, split='test'): #CHANGED THE PATH TO TEST DATA
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = AudioDataset(cfg, split)        # create an object instance of our AudioDataset class
    
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'], #whatever I said was my batch size in the config file, this will be represented here bc it is pulled from that file 
            shuffle=True, #shuffles files read in, which changes them every epoch. Usually turned off for train and val sets. Must do manually. 
            num_workers=cfg['num_workers']
        )
    return dataLoader


# This tells us how to start a model that we have previoussly stopped or paused, and we need to start from the same epoch 
def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = BeeNet(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('model_states/*.pt') #this looks for the saved model files
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states] #you can define what epoch you want to start on 
        eval_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Evaluating from epoch {eval_epoch}')
        state = torch.load(open(f'model_states/{eval_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

        # MOST OF THIS WILL BE COMMENTED OUT FOR US BECAUSE WE WONT BE STARTING WITH SOMETHING NEW

    else:
        # no save state found; start anew
        print('No model found')
        #start_epoch = 0

    return model_instance, eval_epoch


# THIS IS HOW THE MODEL TRAINS. The validation function is almost the same, some key differences: no backward pass here. We do not run the optimizer here: optimize
# on the training data, but not validate. 
def predict(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()

    # # iterate over dataLoader
    # progressBar = trange(len(dataLoader))
    
    with torch.no_grad(): 
        true_labels = []
        predicted_labels = []
        confidences_0 = [] 
        confidences_1 = confidence[:,1]
        confidences_2 = confidence[:,2]
        confidences_3 = confidence[:,3]
        confidences_4 = confidence[:,4]
        confidences_5 = confidence[:,5]
        confidences_6 = confidence[:,6]
        confidences_7 = confidence[:,7]
        confidences_8 = confidence[:,8]             
        
        # to - do: add individual  
        # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)
            true_labels.extend(labels)

            # forward pass
            prediction = model(data)

            pred_label = torch.argmax(prediction, dim=1).numpy()
            predicted_labels.extend(pred_label)

            confidence = torch.nn.Softmax(dim=1)(prediction).numpy() #this is a long confidence probability vector 
            confidence_0 = confidence[:,0]
            confidence_1 = confidence[:,1]
            confidence_2 = confidence[:,2]
            confidence_3 = confidence[:,3]
            confidence_4 = confidence[:,4]
            confidence_5 = confidence[:,5]
            confidence_6 = confidence[:,6]
            confidence_7 = confidence[:,7]
            confidence_8 = confidence[:,8]

            confidences_0.extend(confidence_0) 

    return predicted_labels, true_labels, confidences_0, confidences_1, confidences_2, confidences_3, confidences_4, confidences_5, confidences_6, confidences_7, confidences_8


def save_confusion_matrix(true_labels, predicted_labels, cfg):
    # make figures folder if not there

    matrix_path = cfg['data_root']+'/figs'
    #### make the path if it doesn't exist
    if not os.path.exists(matrix_path):  
        os.makedirs(matrix_path, exist_ok=True)

    confmatrix = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confmatrix)
    #confmatrix.save(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_'+ str(split) +'.png', facecolor="white")
    disp.plot()
    plt.savefig(cfg['data_root'] +'/figs/confusion_matrix.png', facecolor="white")
       ## took out epoch)
    return confmatrix

# When you call train.py, Main is what starts running. It has all of the functions we defined above, and it puts them all in here. When you call the file, it looks 
# through Main, and then when it hits a function, it goes to the to the top to understand the function, and then goes back to Main. 
def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.') # this is how it knows to look at different things 
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml') # change path to config using https://github.com/CV4EcologySchool/audio_classifier_example and scrolling down. Change on command line when you run. 
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    dl_val = create_dataloader(cfg, split='test') #dl_val means dataloader validation 

    # initialize model
    model, current_epoch = load_model(cfg)

    predicted_labels, true_labels, confidences_0, confidences_1, confidences_2, confidences_3, confidences_4, confidences_5, confidences_6, confidences_7, confidences_8 = predict(cfg, dataLoader = dl_val, model)  
        
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg)
    print("confusion matrix saved")

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()