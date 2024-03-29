# Query-Guided Resolution in Uncertain Databases
### Authers: Osnat Drien, Matanya Freiman, Antoine Amarilli, Yael Amsterdamer

We present a modular end-to-end framework to manage uncertain data management in a probabilistic database. We develop an end-to-end novel framework solution for resolving Uncertain Databases using guided queries. Our solution helps to avoid complete cleaning of the entire database but clean the relevant tuples of the query result. 

## Data

During the experiments, we used two benchmarks. The first is the [NELL](https://dl.acm.org/doi/10.1145/3191513) dataset, and the second dataset is [TPC-H](http://www.tpc.org/tpch/).
Our framework's data is already processed into the Boolean expressions we try to evaluate. We provide the NELL dataset which is available [here](https://drive.google.com/drive/folders/1deY_M52Vj45qr0Zudzhc0FsOTwxZjqOs?usp=sharing). Please download the archive and place the directories/files in the project's root directory.

In addition, we use a trained model of [LAL](https://proceedings.neurips.cc/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf)(Learning Active Learning). Please download the [trained model](https://github.com/ksenia-konyushkova/LAL/blob/master/lal%20datasets/LAL-iterativetree-simulatedunbalanced-big.npz) (LAL-iterativetree-simulatedunbalanced-big.npz) from its repository under 'lal datasets' directory, and place it in the project's root directory.

## Installation

This project is running on Windows, Python 3.7

Clone this repository to the home directory
```shell
git clone https://github.com/osnatairy/Active-Probabilistic-Databases-SIGMOD-2023.git
```
Next, please install the following packages that are used in our framework:

[NumPy](https://numpy.org/install/) Version: 1.23.3
```shell
pip install numpy
```
[Pandas](https://pandas.pydata.org/docs/getting_started/install.html) Version: 1.4.4
```shell
pip install pandas
```
[scikit-learn](https://scikit-learn.org/stable/install.html) Version: 1.4.0
```shell
pip install -U scikit-learn
```
[boolean](https://pypi.org/project/boolean/) Version: 1.1.0
```shell
pip install boolean
```

## Getting Started
The framework gets three parameters: algorithm, query, and repository size. The values of the different parameters can be chosen from the following values:

**algorithm:** RO_Algorithm, General_Algorithm, Q_Value_Algorithm 

This parameter defines the algorithm for choosing the next probe at each round.

**query:** 'Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8' 

This parameter defines the query we want to run. An explanation of each query appears [here](https://github.com/osnatairy/Active-Probabilistic-Databases-SIGMOD-2023/blob/main/NELL%20Queries.pdf) 

**size_of_repos:** 80, 320, 1280, 5120 

This parameter defines the initial known probes in the repository.

Next, you run: **python main.py algorithm query size_of_repo**, for example:

```bash
python main.py RO_Algorithm Q1 80
```

## Results
The experiments results are held on a file with the name of "\Results_NELL_experiment_initials_{}_file.csv" under NELL folder and a path based on the parameter you choose, for example: NELL/Q1_Result/80_initials
That file contains the running results which are the number of probes it took to evaluate the chosen query.


## Additional Info. File:
On the project's root path, there is a file name: ‘NELL Queries.pdf’, which explains the queries we used from the dataset.
