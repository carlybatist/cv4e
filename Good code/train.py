'''
    Training script. Here, we load the training and validation datasets (and
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))
from sklearn.metrics import precision_score, recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# let's import our own classes and functions!
from util import init_seed
from dataset import AudioDataset
from model import CarNet


def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = AudioDataset(cfg, split)        # create an object instance of our AudioDataset class

    # print(dataset_instance[0])
    
    do_shuffle = split == 'train'
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=do_shuffle,                                   # should only be put on for training
            num_workers=cfg['num_workers']
        )
    return dataLoader



def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CarNet(cfg['num_classes'])         # CHAGNED ; create an object instance of our model class

    # load latest model state
    model_states = glob.glob('testing_metrics_model/*.pt')
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('testing_metrics_model/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'testing_metrics_model/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])
        #f'{}/{start_epoch}.pt'.format(cfg['save_dir']) #empty expression not allowed
        #f'model_states/{start_epoch}.pt'

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('testing_metrics_model', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'testing_metrics_model/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    # cfpath = cfg['save_dir']+'/'+'config.yaml'
    # if not os.path.exists(cfpath):
    #     with open(cfpath, 'w') as f:
    #         yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):                                        ## Adam optimizer?
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    
    device = cfg['device']

    # put model on device
    model.to(device)                                        # device defined in config file
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    #tensor of negative, then positive class weights
    # class_weights = torch.tensor(0.00001603720632, 0.00314465408805).cuda()
    # print(class_weights)
    # print(type(class_weights))

    # loss function (within function, include weight=class_weights)
    criterion = nn.CrossEntropyLoss()

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    pred_labels = []
    pred_scores = []
    true_labels = []

    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics                                # the predicted label is the one at position (class index) with highest predicted value
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor                                                
            
        pred_label = torch.argmax(prediction, dim=1)    # #argmax - turn probability score into binary label - pred_label is tensor of 0's & 1's
        pred_score = torch.softmax(prediction, dim=1)    
        pred_scores.extend(pred_score.cpu().detach().numpy())
        pred_labels.extend(pred_label.cpu().detach().numpy())
        true_labels.extend(labels.cpu().numpy())

        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)

    # end of epoch; finalize
    progressBar.close()
    
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)
    # print(type(true_labels))
    # print(type(pred_labels))
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_scores = np.array(pred_scores)
    
    acc_0 = np.mean(true_labels[true_labels==0] == pred_labels[true_labels==0])
    acc_1 = np.mean(true_labels[true_labels==1] == pred_labels[true_labels==1])
    # print(np.shape(pred_labels))
    bas = balanced_accuracy_score(true_labels, pred_labels)
    
    print(f'bas: {bas}, acc_0: {acc_0}, acc_1: {acc_1}')

    num_pos = np.where(true_labels==1)[0].shape[0]
    pos_weight = 1/num_pos
    num_neg = 1/np.where(true_labels==0)[0].shape[0]
    neg_weight = 1/num_neg
    # print(num_pos)
    # print(pos_weight)
    # print(num_neg)
    # print(neg_weight)

    return loss_total, oa_total


def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    pred_labels = []
    pred_scores = []
    true_labels = []

    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics                                # the predicted label is the one at position (class index) with highest predicted value
            loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor                                                
            
            pred_label = torch.argmax(prediction, dim=1)    # argmax - turn probability score into binary label - pred_label is tensor of 0's & 1's
            pred_score = torch.softmax(prediction, dim=1)    
            pred_scores.extend(pred_score.cpu().detach().numpy())
            pred_labels.extend(pred_label.cpu().detach().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()

    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_scores = np.array(pred_scores)

    acc_0 = np.mean(true_labels[true_labels==0] == pred_labels[true_labels==0])
    acc_1 = np.mean(true_labels[true_labels==1] == pred_labels[true_labels==1])
    bas = balanced_accuracy_score(true_labels, pred_labels)

    print(f'bas: {bas}, acc_0: {acc_0}, acc_1: {acc_1}')
    
    return loss_total, oa_total



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')              ##  WHAT?? 
    parser.add_argument('--config', help='Path to config file', default='config.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    
    #tensorboard initialize
    writer=SummaryWriter()
    
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)

        
        #tensorboard
        writer.add_scalar('Train loss',loss_train,current_epoch)
        writer.add_scalar('Val loss',loss_val,current_epoch)
        writer.add_scalar('Train Accur',oa_train,current_epoch)
        writer.add_scalar('Val Accur',oa_val,current_epoch)
        writer.flush()

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
        }
        save_model(cfg, current_epoch, model, stats)
        #print(stats)

        #tensorboard ends
        writer.close()



if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()