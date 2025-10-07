import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_clam import ABMIL, CARP3D_Naive, CARP3D_LD, CARP3D_LD_Ave, CARP3D_LD_Linear_Attn, CARP3D_LD_RNN, CARP3D_DL
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = 0
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        from focal_loss import FocalLoss
        loss_fn = FocalLoss(gamma=0.5)
    else:
        # cls_num_list = [len(train_split.slide_cls_ids[c]) for c in range(len(train_split.slide_cls_ids))]
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.BCEWithLogitsLoss() #bce loss
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "feat_dim": args.feat_dim}
    
    if args.model_type in ['abmil', 'carp3d_naive', 'carp3d_ld', 'carp3d_ld_ave', 'carp3d_ld_linear_attn', 'carp3d_ld_rnn', 'carp3d_dl']:
        
        if args.model_type =='abmil':
            model = ABMIL(**model_dict)
        elif args.model_type == 'carp3d_naive':
            model = CARP3D_Naive(**model_dict)
        elif args.model_type == 'carp3d_ld':
            model = CARP3D_LD(**model_dict)
        elif args.model_type == 'carp3d_ld_ave':
            model = CARP3D_LD_Ave(**model_dict)
        elif args.model_type == 'carp3d_ld_linear_attn':
            model = CARP3D_LD_Linear_Attn(**model_dict)
        elif args.model_type == 'carp3d_ld_rnn':
            model = CARP3D_LD_RNN(**model_dict)
        elif args.model_type == 'carp3d_dl':
            model = CARP3D_DL(**model_dict)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        raise Exception("Model not defined.")
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True) #early stop

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break
        # Test model performances every 25 epochs to avoid over fitting.
        # Used when leave one out cross validation without validation set for model selection.
        if (epoch+1)%25 == 0 and epoch+1 != args.max_epochs:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_e_{}_checkpoint.pt".format(cur, epoch+1)))
            results_dict, test_error, acc_logger = summary(model, test_loader, args.n_classes)
            print('Epoch {:} Val error: {:.4f}'.format(epoch, test_error))
            filename = os.path.join(args.results_dir, 'split_{}_epoch_{}_results.pkl'.format(cur, epoch+1))
            from utils.file_utils import save_pkl
            save_pkl(filename, results_dict)

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, _= summary(model, val_loader, args.n_classes) 
    print('Val error: {:.4f}'.format(val_error))

    results_dict, test_error, acc_logger, retrun_features = summary(model, test_loader, args.n_classes, return_features=True)#comment for return features
    filename = os.path.join(args.results_dir, 'split_{}_slides.pkl'.format(cur))
    from utils.file_utils import save_pkl
    save_pkl(filename, retrun_features)
    print('Test error: {:.4f}'.format(test_error))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.close()
    return results_dict, 1-test_error, 1-val_error 


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   #weight, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        label = label.to(device)
        logits, Y_prob, Y_hat,  _, _ = model(data)

        acc_logger.log(Y_hat, label)    

        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
           
        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            
            label = label.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}'.format(val_loss, val_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur))) # early_stopping
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def test(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            
            label = label.to(device, non_blocking=True)
            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)
    
    if writer:
        writer.add_scalar('test/loss', val_loss, epoch)
        writer.add_scalar('test/error', val_error, epoch)

    print('\nTest Set, test_loss: {:.4f}, test_error: {:.4f}'.format(val_loss, val_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    return False


def summary(model, loader, n_classes, return_features=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    patient_features = {}

    for batch_idx, (data, label) in enumerate(loader):
        label = label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if return_features:
                logits, Y_prob, Y_hat, _, features = model(data, return_features=True)
                # filename = os.path.join("slide_features", 'split_{}_epoch_{}_results.pkl'.format(cur, epoch+1))
                # from utils.file_utils import save_pkl
                # save_pkl(filename, features)
            else:
                logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy() 
        
        # probs = logits.cpu().numpy() #bce loss
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        if return_features:
            patient_features.update({slide_id: features})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if return_features:
        return patient_results, test_error, acc_logger, patient_features
    else:
        return patient_results, test_error, acc_logger
