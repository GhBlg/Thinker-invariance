import numpy as np
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.windowers import create_windows_from_events
from tqdm import tqdm
from braindecode.models import SleepStagerChambon2018, TIDNet
from torchsummary import summary
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

subject_id = 4
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])


########################################################



trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to define how
# trials should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)


splitted = windows_dataset.split('session')
train_set = splitted['session_T']
valid_set = splitted['session_E']

train_x=np.array([ele[0][:-1] for ele in train_set])
train_y=np.array([ele[1] for ele in train_set])

valid_x=np.array([ele[0][:-1] for ele in valid_set])
valid_y=np.array([ele[1] for ele in valid_set])


############################################################################

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'

############################################################################
class TrainObject(object):
    def __init__(self, X, y):
        assert len(X) == len(y)
        mean = np.mean(X, axis=2, keepdims=True)
        # Here standardize across the window, when channel size is not large enough
        # In motor imagery kit, we put axis = 1, across channel as an example
        std = np.std(X, axis=2, keepdims=True)
        X = (X - mean) / std
        # we scale it to 1000 as a better training scale of the shallow CNN
        # according to the orignal work of the paper referenced above
        self.X = X*1e3
        self.y = y

# create pytorch datasets
class EEGDataset(Dataset):
    def __init__(self, X, labels=None, transforms=None):
        self.X = X
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i,:,:]
                
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data



############################################################################
lim=int(train_x.shape[0]* 0.8)   # 80% training+validation / 20% testing


test_x=np.concatenate((train_x[lim:],valid_x[lim:]))
test_y=np.concatenate((train_y[lim:],valid_y[lim:]))

train_set = TrainObject(train_x[:lim], y=train_y[:lim])
valid_set = TrainObject(valid_x[:lim], y=valid_y[:lim])
test_set = TrainObject(test_x, y=test_y)

train_data = EEGDataset(train_set.X, train_set.y, transforms=None)
valid_data = EEGDataset(valid_set.X, valid_set.y, transforms=None)
test_data = EEGDataset(test_set.X, test_set.y, transforms=None)

############################################################################

def build_network():
    
    #model = SleepStagerChambon2018(n_channels=2, sfreq=100,n_classes=6,apply_batch_norm=True,
                 #time_conv_size_s=1,
                 #dropout=0.2,  max_pool_size_s=0.2 , n_conv_chs=16 ,pad_size_s=0.25)

    model=TIDNet(n_classes=4, in_chans=25, input_window_samples=1125, s_growth=24, t_filters=32,
                 drop_prob=0.4, pooling=15, temp_layers=2, spat_layers=2, temp_span=0.05,
                 bottleneck=3, summary=-1)
    summary(model.cuda(), (25, 1125))
    return model.to(device)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9, weight_decay=0.5*0.001)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(network.parameters(),
                               lr=learning_rate,weight_decay=0.5*0.001, amsgrad=True)
    return optimizer


def train_epoch(network, loader, optimizer, loss_config, batch_size):
    cumu_loss = 0
    correct = 0.0
    total = 0.0
    
    for i, (data, target) in tqdm(enumerate(loader), ncols = 100, total=int(len(train_set.X)/batch_size)+1,
               desc ="Training"):
        data, target = data.to(device), target.to(device)
        if data.shape[0]==batch_size:

            #data=torch.unsqueeze(data,3)
            data.double()
            target.long()
            optimizer.zero_grad()
            network.double()

            # ➡ Forward pass
            if loss_config == "nll_loss":
                #loss = floss.forward(network(data.double()), target.long())
                loss = F.nll_loss(network(data.double()), target.long())
                cumu_loss += loss.item()
            elif loss_config =='CrossEntropyLoss':
                loss = F.cross_entropy(network(data.double()), target.long())
                cumu_loss += loss.item()
        
            # ⬅ Backward pass + weight update
            loss.backward()
            optimizer.step()
        
             # compute accuracy
            outputs = network(data.double())

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += target.size(0)
            correct += (predicted == target).sum()

            #wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader), correct/total

def validate_epoch(network, loader, optimizer, loss_config, batch_size):
    cumu_loss = 0.0
    correct = 0.0
    total = 0.0
   
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        if data.shape[0]==batch_size:
            data.double()
            target.long()
            
        
            optimizer.zero_grad()
            network.double()
            network.eval()  
            torch.no_grad()
            
            # Compute loss
            if loss_config == "nll_loss":
                loss = floss.forward(network(data.double()), target.long())
                #loss = F.nll_loss(network(data.double()), target.long())
                cumu_loss += loss.item()
            elif loss_config =='CrossEntropyLoss':
                loss = F.cross_entropy(network(data.double()), target.long())
                cumu_loss += loss.item()      

          # compute accuracy
            outputs = network(data.double())

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += target.size(0)
            correct += (predicted == target).sum()   

    return cumu_loss / len(loader), correct/total


def test(network, loader, batch_size, n_classes):
    # Calculate Accuracy
    correct = 0.0
    correct_arr = [0.0] * n_classes
    total = 0.0
    total_arr = [0.0] * n_classes
    y_true=[]
    y_pred=[]
    # Iterate through test dataset
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        if data.shape[0] == batch_size:  #condition to avoid taking trials length < batch size (problematic for confusion matrix)
            data.double()
            target.long()
            network.double()
            outputs = network(data)
                    # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            # Total number of labels
            total += target.size(0)
            correct += (predicted == target).sum()
            y_true.append(target)
            y_pred.append(predicted)
           
            for label in range(n_classes):
                correct_arr[label] += (((predicted == target) & (target==label)).sum())
                total_arr[label] += (target == label).sum()

    accuracy = correct / total
    print('TEST ACCURACY {} '.format(accuracy))
    
    print('TEST F1-Score {} '.format(f1_score(torch.tensor(np.array(y_true,'int32')).view(-1), torch.tensor(np.array(y_pred,'int32')).view(-1),  average='macro')))
            
    return accuracy


def train_no_wandb(config=None):
    n_classes=4
    epochs=10
    batch_size=5
    LR=1e-3
    network = build_network()
    optimizer = build_optimizer(network, 'sgd', LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

    Train_acc=[]
    Val_acc=[]
    Train_loss=[]
    Val_loss=[]
    Test_acc=[]

    for epoch in range(epochs):#config.epochs
        train_loss,train_acc = train_epoch(network, train_loader, optimizer, 'CrossEntropyLoss', batch_size)
        print('train loss {} accuracy {} epoch {} done'.format(train_loss,train_acc,epoch))
        val_loss,val_acc = validate_epoch(network, valid_loader, optimizer,'CrossEntropyLoss', batch_size)
        print('val loss {} epoch {} done'.format(val_loss,epoch))
        Train_acc.append(train_acc)
        Val_acc.append(val_acc)
        Train_loss.append(train_loss)
        Val_loss.append(val_loss)
        scheduler.step(val_loss)
        if epoch % 1 == 0:
            test_acc=test(network, test_loader, batch_size, n_classes)
            Test_acc.append(test_acc)

    return Train_acc, Val_acc, Test_acc

############################################################################

Train_acc, Val_acc, Test_acc=train_no_wandb()
