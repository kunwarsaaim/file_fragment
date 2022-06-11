# Data Carving Research

This research imporoves upon previous paper called "FiFTy: Large-scale File Fragment Type Identification using Neural Networks" which performs file type classification based on. The original paper can be accessed at [FiFTy Paper](https://arxiv.org/abs/1908.06148). The Datasets used in this research can be accessed through the following link [FFT-75](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset).

## Training

First, download all the dataset, decompress the folders and put them in dataset folder.
`data/<dataset folders>`

## RUN Script
Run this script for inference time

```python
!python inference_time.py --data_folder ./data/ --results_folder ./results/
```

### Train

```bash
python train.py --data_folder ./data/ --results_folder ./results/ --scenario 4 --size 4096 --batch_size 128 --lr 0.002
```

This command will start a training session using the data under the *data/size_scenario* directory. See `./train --help` for a description of command line arguments.

The ressults are saved under the *results/scenario/size* directory.

## Evaluation

```bash
python eval.py --data_folder ./data/ --results_folder ./results/ --scenario 0
```

This command will load previously trained model for each scenario [1, 6] and evaluate the model on the validation dataset.

It will then generate confusion matrix images and store them in eps format under the results folder.

The results directory will be structured as following:

 ```bash
 ├── results
    |   ├── 1
    |   |   ├── 4096
    |   |   └── 512
    |   |
    |   ├── 2
    |   |   ├── 4096
    |   |   └── 512
    |   |
    |   |── 3
    |   |   ├── 4096
    |   |   └── 512
    |   |
    |   ├── 4
    |   |   ├── 4096
    |   |   └── 512
    |   |
    |   ├── 5
    |   |   ├── 4096
    |   |   └── 512
    |   |
    |   ├── 6
    |       ├── 4096
    |       └── 512
 ```

## 2. Requirements

```bash
python>=3.6
torch>=1.5.0
torchvision
torchsummary
numpy
scipy
matplotlib
```
