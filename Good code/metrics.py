from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve

# classification report (Ethan's suggestion)
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))


## FROM CATHERINE

def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = AudioDataset(cfg, split)        # create an object instance of our AudioDataset class

    # print(dataset_instance[0])
    # assert 0
    
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
    model_instance = CarNet(cfg['num_classes'])         # CHANGED ; create an object instance of our model class

    # load latest model state
    model_states = glob.glob('model_states_batch128/*.pt')
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states_batch128/','').replace('.pt','')) for m in model_states]
        eval_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states_batch128/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        eval_epoch = 0

    return model_instance, eval_epoch

true_labels = labels
predicted_labels = pred_score                                                                    ## pred_score vs pred_label ?

def save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch='32', split='train'):    # epoch?? args??
    
    # make figures folder if not there
    matrix_path = cfg['data_root']+'/experiments/'+(args.exp_name)+'/figs'
    #### make the path if it doesn't exist
    if not os.path.exists(matrix_path):                                                     # matrix_path??
        os.makedirs(matrix_path, exist_ok=True)

    confmatrix = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confmatrix)
    #confmatrix.save(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_'+ str(split) +'.png', facecolor="white")
    disp.plot()
    plt.savefig(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_'+ str(epoch) +'.png', facecolor="white")
       ## took out epoch)
    return confmatrix

## we will calculate overall precision, recall, and F1 score
#def save_accuracy_metrics(y_true, y_pred, args, epoch, split):

    # make a csv of accuracy metrics 

def save_precision_recall_curve(true_labels, predicted_labels, cfg, args, epoch='128', split='train'):
        #### make the path if it doesn't exist
    if not os.path.exists('experiments/'+(args.exp_name)+'/figs'):
        os.makedirs('experiments/'+(args.exp_name)+'/figs', exist_ok=True)
    
    PRcurve = PrecisionRecallDisplay.from_predictions(true_labels, predicted_labels)
    PRcurve.plot()
    plt.savefig(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/PRcurve'+str(epoch)+'_'+ str(split) +'.png', facecolor="white")


# get accuracy score
    ### this is just a way to get two decimal places 
    acc = accuracy_score(true_labels, predicted_labels)
    print("Accuracy of model is {:0.2f}".format(acc))

    # confusion matrix
    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg, args, epoch = epoch, split = 'train')
    print("confusion matrix saved")

    if cfg['num_classes'] == 2:
        ######################### put this all in a function ##############
        # get precision score
        ### this is just a way to get two decimal places 
        precision = precision_score(true_labels, predicted_labels)
        print("Precision of model is {:0.2f}".format(precision))

        # get recall score
        ### this is just a way to get two decimal places 
        recall = recall_score(true_labels, predicted_labels)
        print("Recall of model is {:0.2f}".format(recall))

        # get recall score
        ### this is just a way to get two decimal places 
        F1score = f1_score(true_labels, predicted_labels)
        print("F1score of model is {:0.2f}".format(F1score))
        ######################################################################

        PRcurve = save_precision_recall_curve(true_labels, predicted_labels, cfg, args, epoch = epoch, split = 'train')
        print("precision recall curve saved")


