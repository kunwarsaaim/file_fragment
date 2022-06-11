import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json
import os
from models.inception_depth_conv import Separable_Inception_network
from utils import accuracy,get_data
from plot_cf_matrix import plot_confusion_matrix



def evaluate(device, model, results_folder, scenario, size, data_folder,epoch):
    pth_path = os.path.join(results_folder, str(scenario), str(size), "inception_depth_conv_epoch_"+str(epoch)+".pth")
    if device == torch.device('cuda'):
        checkpoint = torch.load(pth_path)
    else:
        checkpoint = torch.load(pth_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    accuracy_batch = 0.0
    prediction = []
    _, eval_data_loader = get_data(scenario=scenario, batch_size=256, data_folder=data_folder, mode='test', size=size)
    with torch.no_grad():
        for i, data in enumerate(eval_data_loader, 0):
            x, y = data
            if device == torch.device('cuda'):
                x = Variable(x).long().cuda()
                y = Variable(y).cuda()
            else:
                x = Variable(x).long()
                y = Variable(y)
            y_hat = model(x)
            accuracy_batch += accuracy(y, y_hat)
            prediction.append(y_hat.argmax(1))
    eval_accuracy = accuracy_batch / len(eval_data_loader)
    print('====> Eval Accuracy for Scenario ' + str(scenario) + ' Size ' + str(size) + ': {:.2f}'.format(eval_accuracy))

    cf_matrix = confusion_matrix(eval_data_loader.dataset.tensors[1].float().detach().cpu().numpy(), torch.cat(prediction).float().detach().cpu().numpy())

    with open('labels.json') as f:
        data = json.load(f)
    class_name = data[str(scenario)]
    figsizes = [(35, 35), (12, 12), (15, 15), (6, 6), (2.5, 2.5), (2.5, 2.5)]
    sz = figsizes[scenario-1]
    fig, ax = plot_confusion_matrix(conf_mat=cf_matrix,
                                show_absolute=False,
                                show_normed=True,
                                cmap = 'binary',
                                class_names=class_name,
                                figsize=sz)
    plt.title("Scenario: " + str(scenario) + "\nFragment size: " + str(size) + "\nAccuracy:{:.2f}".format(eval_accuracy))
    plt.savefig(os.path.join(results_folder, 'Confusion_matrix_Scenario_' + str(scenario) + '_Size_' + str(size) + '.eps'), format='eps', bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="data", help='Folder contating FFT-75 Datasets [Default: data]')
    parser.add_argument('--results_folder', type=str, default="results", help='Results Folder, results will be written to subdirectories of this folder [Default: results]')
    parser.add_argument('--scenario', type=int, default=0, help='which subset of dataset(1 to 6) [Default: 1] or 0 (get evaluation of all Scenario)')
    parser.add_argument('--size', type=int, default=4096, help='size of fragment(4096 or 512) [Default: 4096]')

    opt = parser.parse_args()

    data_folder = opt.data_folder
    results_folder = opt.results_folder
    scenario = opt.scenario
    size = opt.size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluation on GPU") if device == torch.device('cuda') else print('Evaluation on CPU')
    classes = [0, 75, 11, 25, 5, 2, 2] #number of different type of files predicted in each scenario
    pool_size = 4

    epochs = [[18,18,19,19,18,18],
        [18,18,19,18,19,18]]


    if scenario == 0:
        for i in range(6):
            class_ = classes[i+1]
            if device ==  torch.device('cuda'):
                model = Separable_Inception_network(class_, pool_size).cuda()
            else:
                model = Separable_Inception_network(class_, pool_size)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)

            evaluate(device, model, results_folder, i+1, 4096, data_folder,epochs[0][i])
            evaluate(device, model, results_folder, i+1, 512, data_folder,epochs[1][i])
    else:
        class_ = classes[scenario]
        if device ==  torch.device('cuda'):
            model = Separable_Inception_network(class_, pool_size).cuda()
        else:
            model = Separable_Inception_network(class_, pool_size)
        if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)

        evaluate(device, model, results_folder, scenario,size, data_folder,epochs[0 if size == 4096 else 1][scenario-1])

if __name__ == "__main__":
    main()
