from models.inception_depth_conv import Separable_Inception_network
from utils import accuracy,get_data,pytorch_count_params
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from time import time, gmtime
import sys

def printStats(title, timediff):
    print("{} Time: {:2d}d {:2d}h {:2d}m {:2d}s".format(title, (timediff.tm_mon - 1), timediff.tm_hour, timediff.tm_min, timediff.tm_sec))

def train(epoch, data_loader, training = True, results_folder = "./results/1/4k"):
    accuracy_batch = 0.0
    train_loss = 0.0
    model.train()
    for i, data in enumerate(data_loader, 0):
        x,y = data
        if device == torch.device('cuda'):
            x = Variable(x).long().cuda()
            y = Variable(y).cuda()
        else:
            x = Variable(x).long()
            y = Variable(y)
        optim.zero_grad()
        y_hat = model(x)
        loss = criterion_class(y_hat,y)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        accuracy_batch += accuracy(y,y_hat)
    print('====> Epoch: {} Average train accuracy: {:.6f} Average train loss: {:.6f}'.format(
          epoch+1, accuracy_batch / (len(data_loader)),train_loss/(len(data_loader))))
    if training:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'epoch':epoch
        },os.path.join(results_folder, 'inception_depth_conv_epoch_' + str(epoch) + '.pth'))
    return model

def test(epoch, data_loader):
    accuracy_batch = 0.0
    test_loss = 0.0
    model.eval()
    for i,data in enumerate(data_loader,0):
        x,y = data
        if device == torch.device('cuda'):
            x = Variable(x).long().cuda()
            y = Variable(y).cuda()
        else:
            x = Variable(x).long()
            y = Variable(y)
        y_hat = model(x)
        loss = criterion_class(y_hat,y)
        test_loss += loss.item()
        accuracy_batch += accuracy(y,y_hat)
    print('====> Epoch: {} Average test accuracy: {:.6f} Average test loss: {:.6f}'.format(
          epoch+1, accuracy_batch / (len(data_loader)),test_loss/(len(data_loader))))

def main():
    global model, device, optim, criterion_class
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="data", help='Folder contating FFT-75 Datasets [Default: data]')
    parser.add_argument('--results_folder', type=str, default="results", help='Results Folder, results will be written to subdirectories of this folder [Default: results]')
    parser.add_argument('--scenario', type=int, default=1, help='which subset of dataset(1 to 6) [Default: 1]')
    parser.add_argument('--size', type=int, default=4096, help='size of fragment(4096 or 512) [Default: 4096]')
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini batch [Default: 128]')
    parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate [Default: 0.002]')
    opt = parser.parse_args()


    scenario = opt.scenario
    size= opt.size
    batch_size = opt.batch_size
    lr = opt.lr
    data_folder = opt.data_folder
    results_folder = opt.results_folder
    classes = [0,75,11,25,5,2,2] #number of different type of files predicted in each scenario
    class_ = classes[scenario]
    pool_size = 4   # We'll try different values(2,4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device ==  torch.device('cuda'):
        model = Separable_Inception_network(class_,pool_size).cuda()
        print("Training on GPU")
    else:
        model = Separable_Inception_network(class_,pool_size)
        print('Training on CPU')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print("Number of Parameters:",pytorch_count_params(model))
    criterion_class= nn.NLLLoss() # Negative log likely Loss function
    optim = torch.optim.Adam(model.parameters(),lr=lr) #Adam optimizer

    train_dataloader_4096, test_dataloader_4096, train_dataloader_512, test_dataloader_512 = get_data(scenario, batch_size, data_folder=data_folder, mode="train", size=size)

    result_folder = os.path.join(results_folder, str(scenario), str(size))
    os.makedirs(result_folder, exist_ok=True)
    pre_training_train_time = 0.0
    pre_training_test_time = 0.0
    training_time = 0.0
    testing_time = 0.0
    if size==4096:
        print("========================Pretraining=========================")
        print("Dataset({}_{}): train={}, test={}".format(512, scenario, len(train_dataloader_512.dataset), len(test_dataloader_512.dataset)))
        for epoch in range(20):
            start = time()
            train(epoch, train_dataloader_512, False)
            end = time()
            pre_training_train_time += (end - start)
            start = time()
            test(epoch, test_dataloader_512)
            end = time()
            pre_training_test_time += (end - start)
        print("========================Training============================")
        print("Dataset({}_{}): train={}, test={}".format(size, scenario, len(train_dataloader_4096.dataset), len(test_dataloader_4096.dataset)))
        for epoch in range(20):
            start = time()
            train(epoch, train_dataloader_4096, True, results_folder = result_folder)
            end = time()
            training_time += (end - start)
            start = time()
            test(epoch, test_dataloader_4096)
            end = time()
            testing_time += (end - start)
    else:
        print("========================Pretraining=========================")
        print("Dataset({}_{}): train={}, test={}".format(4096, scenario, len(train_dataloader_4096.dataset), len(test_dataloader_4096.dataset)))
        for epoch in range(5):
            start = time()
            train(epoch, train_dataloader_4096, False)
            end = time()
            pre_training_train_time += (end - start)
            start = time()
            test(epoch, test_dataloader_4096)
            end = time()
            pre_training_test_time += (end - start)
        print("========================Training============================")
        print("Dataset({}_{}): train={}, test={}".format(size, scenario, len(train_dataloader_512.dataset), len(test_dataloader_512.dataset)))
        for epoch in range(20):
            start = time()
            train(epoch, train_dataloader_512, True, results_folder = result_folder)
            end = time()
            training_time += (end - start)
            start = time()
            test(epoch,test_dataloader_512)
            end = time()
            testing_time += (end - start)
    printStats("Pre-Training Train", gmtime(pre_training_train_time))
    printStats("Pre-Training Test", gmtime(pre_training_test_time))
    printStats("Training", gmtime(training_time))
    printStats("Testing", gmtime(testing_time))


if __name__ == "__main__":
    main()
