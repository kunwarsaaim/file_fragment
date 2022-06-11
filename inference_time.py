from models.inception_depth_conv import Separable_Inception_network
from utils import accuracy,get_data
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from time import time, gmtime
import sys
import itertools
import copy


def inference_time_per_block(device, model, results_folder, scenario, size, data_folder,epoch):
    pth_path = os.path.join(results_folder, str(scenario), str(size), "inception_depth_conv_epoch_"+str(epoch)+".pth")
    if device == torch.device('cuda'):
        checkpoint = torch.load(pth_path)
    else:
        checkpoint = torch.load(pth_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, eval_data_loader = get_data(scenario=scenario, batch_size=1, data_folder=data_folder, mode='test', size=size)

    x = eval_data_loader.dataset.tensors[0][:1]

    if device == torch.device('cuda'):
         x = Variable(x).long().cuda()
    else:
        x = Variable(x).long()

    with torch.no_grad():
        total_time = 0.0
        y_hat = model(x)
        for i in range(10):
            start = time()
            y_hat = model(x)
            end = time()
            total_time+= end - start
        total_time/=10.0
        print("Inference time for Scenario "+str(scenario)+" Size "+str(size)+" is:",str(total_time))


def inference_per_gb_for_scenario_1(device,model,results_folder,data_folder,epoch):

    pth_path_4096 = os.path.join(results_folder, str(1), str(4096), "inception_depth_conv_epoch_"+str(epoch)+".pth")
    pth_path_512 = os.path.join(results_folder, str(1), str(512), "inception_depth_conv_epoch_"+str(epoch)+".pth")

    if device == torch.device('cuda'):
        checkpoint_4096 = torch.load(pth_path_4096)
    else:
        checkpoint_4096 = torch.load(pth_path_4096,map_location=torch.device('cpu'))

    if device == torch.device('cuda'):
        checkpoint_512 = torch.load(pth_path_512)
    else:
        checkpoint_512 = torch.load(pth_path_512,map_location=torch.device('cpu'))

    model_4096 = copy.deepcopy(model)
    model_512 = copy.deepcopy(model)

    model_4096.load_state_dict(checkpoint_4096['model_state_dict'])
    model_4096.eval()

    model_512.load_state_dict(checkpoint_512['model_state_dict'])
    model_512.eval()

    train_dataloader_4096, test_dataloader_4096, train_dataloader_512, test_dataloader_512 = get_data(scenario=1,batch_size=256, data_folder=data_folder, mode="train", size=4096)
    
    total_time = 0.0
    with torch.no_grad():
        for x,y in itertools.islice(train_dataloader_4096,1024):
            if device == torch.device('cuda'):
                x = Variable(x).long().cuda()
            else:
                x = Variable(x).long()
            start = time()
            y_hat = model_4096(x)
            end = time()
            total_time+= end-start
        print('Inference time for scenario 1 and size 4096 for 1gb data is:',total_time,'seconds')

    total_time = 0.0
    with torch.no_grad():
        for x,y in itertools.islice(train_dataloader_512,8192):
            if device == torch.device('cuda'):
                x = Variable(x).long().cuda()
            else:
                x = Variable(x).long()
            start = time()
            y_hat = model_512(x)
            end = time()
            total_time+= end-start
        print('Inference time for scenario 1 and size 512 for 1gb data is:',total_time,'seconds')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="data", help='Folder contating FFT-75 Datasets [Default: data]')
    parser.add_argument('--results_folder', type=str, default="results", help='Results Folder, results will be written to subdirectories of this folder [Default: results]')
    opt = parser.parse_args()
    classes = [0, 75, 11, 25, 5, 2, 2] #number of different type of files predicted in each scenario
    pool_size = 4

    epochs = [[18,18,19,19,18,18],
        [18,18,19,18,19,18]]

    device = torch.device('cuda')
    print("Using GPU")
    for i in range(6):
        class_ = classes[i+1]
        model = Separable_Inception_network(class_, pool_size).cuda()

        inference_time_per_block(device,model,opt.results_folder,i+1,4096,opt.data_folder,epochs[0][i])
        inference_time_per_block(device,model,opt.results_folder,i+1,512,opt.data_folder,epochs[1][i])

        if i == 0:
           inference_per_gb_for_scenario_1(device,model,opt.results_folder,opt.data_folder,epoch=epochs[0][i])

    device = torch.device('cpu')
    print('Using CPU')
    for i in range(6):
        class_ = classes[i+1]
        model = Separable_Inception_network(class_,pool_size)

        inference_time_per_block(device,model,opt.results_folder,i+1,4096,opt.data_folder,epochs[0][i])
        inference_time_per_block(device,model,opt.results_folder,i+1,512,opt.data_folder,epochs[1][i])

        if i == 0:
            inference_per_gb_for_scenario_1(device,model,opt.results_folder,opt.data_folder,epoch=epochs[0][i])

if __name__ == '__main__':
    sys.stdout=open("inference_time_seperable_conv.txt","w")
    main()
    sys.stdout.close()
