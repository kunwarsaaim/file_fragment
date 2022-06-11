from functools import reduce
import numpy as np
from torch import tensor, int64
from torch.utils.data import TensorDataset,DataLoader
import os

def pytorch_count_params(model):
    "count number trainable parameters in a pytorch model"
    total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params

#Accuracy function
def accuracy(true,pred):
    acc = (true == pred.argmax(1)).float().detach().cpu().numpy()
    return float(100 * acc.sum() / len(acc))


def get_data(scenario, batch_size, data_folder='data', mode='train', size=4096,pin_memory=False):
    if mode == "train":
        data_train = np.load(os.path.join(data_folder,'4k_'+str(scenario),'train.npz')) #load zipped numpy file
        data_test = np.load(os.path.join(data_folder,'4k_'+str(scenario),'test.npz'))

        data_train_512 = np.load(os.path.join(data_folder,'512_'+str(scenario),'train.npz'))
        data_test_512 = np.load(os.path.join(data_folder,'512_'+str(scenario),'test.npz'))

        train_x = tensor(data_train['x']).view(-1,4096)
        train_y = tensor(data_train['y']).long()
        test_x = tensor(data_test['x']).view(-1,4096)
        test_y = tensor(data_test['y']).long()

        train_x_512 = tensor(data_train_512['x']).view(-1,512)
        train_y_512 = tensor(data_train_512['y']).long()
        test_x_512 = tensor(data_test_512['x']).view(-1,512)
        test_y_512 = tensor(data_test_512['y']).long()

        train_dataset = TensorDataset(train_x,train_y)  # create your dataset
        test_dataset = TensorDataset(test_x,test_y)
        train_dataloader_4096 = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=pin_memory)
        test_dataloader_4096 = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,pin_memory=pin_memory)
        
        train_dataset_512 = TensorDataset(train_x_512,train_y_512) 
        test_dataset_512 = TensorDataset(test_x_512,test_y_512)
        train_dataloader_512 = DataLoader(train_dataset_512,batch_size=batch_size,shuffle=True,pin_memory=pin_memory)
        test_dataloader_512 = DataLoader(test_dataset_512,batch_size=batch_size,shuffle=True,pin_memory=pin_memory)

        return train_dataloader_4096,test_dataloader_4096,train_dataloader_512,test_dataloader_512
    elif mode == 'test':
        sz = "4k" if size == 4096 else "512"
        data_test = np.load(os.path.join(data_folder, sz + "_" + str(scenario), 'test.npz')) #load zipped numpy file
        data_val = np.load(os.path.join(data_folder, sz + "_" + str(scenario), 'val.npz'))

        test_x = tensor(data_test['x']).view(-1,size)
        test_y = tensor(data_test['y']).long()
        val_x = tensor(data_val['x']).view(-1,size)
        val_y = tensor(data_val['y']).long()
        test_dataset = TensorDataset(test_x,test_y)  # create your dataset
        val_dataset = TensorDataset(val_x,val_y)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)
        val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,pin_memory=pin_memory)
        return test_dataloader,val_dataloader


