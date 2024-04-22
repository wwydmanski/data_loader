import csv
import os
import requests
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from torch.utils import data
import datasets
from typing import Tuple
import openml
import zipfile
from io import StringIO
import tempfile
import requests
from ucimlrepo import fetch_ucirepo 

def load_dataset(dataset_name) -> Tuple[data.Dataset, data.Dataset]:
    """
    Load dataset. 
    :param dataset_name: name of the dataset. One of:
        - MNIST
        - TUANDROMD
        - BlogFeedback
        - BreastCancer
        - reuters
        - letter
        - ColorectalCarcinoma
        - ColorectalCarcinomaCLR
        - Bioresponse
        - EyeMovements
        - Ionosphere
        - Libras
        - Lymphography
        - OvarianTumour
        - Christine
        - Connectionist
        - Dermatology
        - Glass
        - CNAE9
        - ZOO
        - Sonar
        - Dermatology
        - Glass
        - Adult
        - Helena
        - Parkinsons
        - Haberman
        - Vertebral
        - Ecoli
        - Voting


    :return: train_dataset, test_dataset
    """
    if dataset_name == 'MNIST':
        dataset = datasets.load_dataset("mnist")
        
        X_ = [torch.from_numpy(np.asarray(i)) for i in dataset['train'][:-1]['image']]
        X = torch.stack(X_).to(torch.float32)
        y = [torch.from_numpy(np.asarray(i)) for i in dataset['train'][:-1]['label']]
        y = torch.stack(y)
        
        mods = [transforms.Normalize((X.mean()), (X.std())),    #mean and std of MNIST
            transforms.Lambda(lambda x: torch.flatten(x, 1))]

        mods = transforms.Compose(mods)
        X = mods(X)
        X_train, X_test, y_train, y_test = train_test_split(X.to(float), y.to(int), random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(y_train)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(y_test)).type(torch.FloatTensor))
        return train_dataset, test_dataset

    if dataset_name == 'TUANDROMD':
        CSV_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00622/TUANDROMD.csv'
        with requests.Session() as s:
            download = s.get(CSV_URL)

            decoded_content = download.content.decode('utf-8')

            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            df = pd.DataFrame(my_list[1:],columns=my_list[0])
            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)
            df.loc[(df.Label == 'malware'),'Label']= '1'
            df.loc[(df.Label == 'goodware'),'Label']= '0'
            labels = df.pop('Label').values
            X_train, X_test, Y_train, Y_test = train_test_split(df.astype(float), labels.astype(int), random_state=42)
            train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(Y_train)))
            test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(Y_test)).type(torch.FloatTensor))
            return train_dataset, test_dataset
    
    elif dataset_name == 'BlogFeedback':
        train_data_np = []
        test_data_np = []
        for file in os.listdir('../datasets/BlogFeedback'):
            if "test" in file: 
                with open('../datasets/BlogFeedback/' + file, newline='') as csvfile:
                    cr = csv.reader(csvfile, delimiter=',')
                    my_list_test = list(cr)
                    test_data_np.extend(my_list_test)
            if "train" in file: 
                with open('../datasets/BlogFeedback/' + file, newline='') as csvfile:
                    cr = csv.reader(csvfile, delimiter=',')
                    my_list_train = list(cr)
                    train_data_np.extend(my_list_train)

        train_df = pd.DataFrame(train_data_np)
        test_df = pd.DataFrame(test_data_np)
        train_labels = train_df.pop(280).values
        test_labels = test_df.pop(280).values

        scaler = MinMaxScaler()
        train_df[train_df.columns] = scaler.fit_transform(train_df[train_df.columns])
        test_df[test_df.columns] = scaler.fit_transform(test_df[test_df.columns])

        train_labels[train_labels.astype(float) > 0.0] = 1.0
        train_labels[train_labels.astype(float) == 0.0] = 0.0

        test_labels[test_labels.astype(float) > 0.0] = 1.0
        test_labels[test_labels.astype(float) == 0.0] = 0.0
        
        train_dataset = data.TensorDataset(torch.tensor(np.array(train_df)).type(torch.FloatTensor), torch.tensor(np.array(train_labels).astype(int)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(test_df)).type(torch.FloatTensor), torch.tensor(np.array(test_labels).astype(int)))
        return train_dataset, test_dataset

    elif dataset_name == "BreastCancer":
        X, y = load_breast_cancer(return_X_y=True)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(Y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(Y_test))
       
        return train_dataset, test_dataset

    elif dataset_name == 'reuters':
        ds = datasets.load_dataset("wwydmanski/reuters10k")

        X_train = np.array(ds['train']['features'])
        Y_train = np.array(ds['train']['label'])
        X_test = np.array(ds['test']['features'])
        Y_test = np.array(ds['test']['label'])

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(Y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(Y_test))

        return train_dataset, test_dataset

    elif dataset_name == 'letter':
        LETTER_DATA = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        with requests.Session() as s:
            download = s.get(LETTER_DATA)

            decoded_content = download.content.decode('utf-8')

            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            df = pd.DataFrame(my_list)
            labels = df.pop(0).values
            scaler = MinMaxScaler()
            dataset = scaler.fit_transform(df)

            X_train = dataset[:15000]
            Y_train = labels[:15000]
            X_test = dataset[15000:]
            Y_test = labels[15000:]

            le = LabelEncoder()
            le.fit(Y_train)
            Y_train = le.transform(Y_train)
            Y_test = le.transform(Y_test)

            train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(Y_train).astype(int)))
            test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(Y_test).astype(int)))

            return train_dataset, test_dataset
    elif dataset_name == 'ColorectalCarcinoma':
        dataset = datasets.load_dataset("wwydmanski/colorectal-carcinoma-microbiome-fengq", "presence-absence")
        train_dataset, test_dataset = dataset['train'], dataset['test']

        scaler = StandardScaler()
        X_train = np.array(train_dataset['values'])
        y_train = np.array(train_dataset['target'])
        X_train = scaler.fit_transform(X_train)

        X_test = np.array(test_dataset['values'])
        y_test = np.array(test_dataset['target'])
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(y_train).astype(int)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(y_test).astype(int)))

        return train_dataset, test_dataset
    elif dataset_name == 'ColorectalCarcinomaCLR':
        dataset = datasets.load_dataset("wwydmanski/colorectal-carcinoma-microbiome-fengq", "CLR")
        train_dataset, test_dataset = dataset['train'], dataset['test']
        
        scaler = StandardScaler()
        X_train = np.array(train_dataset['values'])
        y_train = np.array(train_dataset['target'])
        X_train = scaler.fit_transform(X_train)

        X_test = np.array(test_dataset['values'])
        y_test = np.array(test_dataset['target'])
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(np.array(X_train)).type(torch.FloatTensor), torch.tensor(np.array(y_train).astype(int)))
        test_dataset = data.TensorDataset(torch.tensor(np.array(X_test)).type(torch.FloatTensor), torch.tensor(np.array(y_test).astype(int)))

        return train_dataset, test_dataset
    elif dataset_name == "Bioresponse":
        dataset = datasets.load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/Bioresponse.csv")
        df = pd.DataFrame(dataset['train'])
        y = df.pop('target').values
        X = df.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "EyeMovements":
        dataset = datasets.load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/eye_movements.csv")
        df = pd.DataFrame(dataset['train'])
        y = df.pop('label').values
        X = df.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Cleveland":
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", 
            header=None,
            na_values="?"
            ).dropna()
        y = df.pop(13).values
        X = df.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Ionosphere":
        ionosphere = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data', header=None)
        ionosphere = ionosphere.drop(1, axis=1)
        X = ionosphere.values[:, :-1].astype(float)
        y = ionosphere.values[:, -1]
        y = LabelEncoder().fit_transform(y).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, train_size=0.7)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Libras":
        libras = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data', header=None)
        X = libras.values[:, :-1].astype(float)
        y = libras.values[:, -1].astype(int)
        y -= 1
        #(360, 90) 15
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, train_size=0.7)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Lymphography":
        lymphography = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data', header=None)
        X = lymphography.values[:, 1:].astype(float)
        y = lymphography.values[:, 0].astype(int)
        y -= 1
        #(148, 18) 4
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, train_size=0.7)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Sonar":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", header=None)
        X = dataset.values[:, :-1].astype(float)
        y = dataset.values[:, -1]
        y = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, train_size=0.7)
        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Dermatology":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data", header=None, na_values="?").dropna()
        X = dataset.values[:, :-1].astype(float)
        y = dataset.values[:, -1].astype(int)
        y = LabelEncoder().fit_transform(y).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, train_size=0.7)
        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Glass":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data", header=None, na_values="?").dropna()
        X = dataset.values[:, :-1].astype(float)
        y = dataset.values[:, -1].astype(int)
        y = LabelEncoder().fit_transform(y).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, train_size=0.7)
        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "OvarianTumour":
        # dir_path = os.path.dirname(os.path.realpath(__file__))

        # dataset = pd.read_csv(f"{dir_path}/tumour.csv")
        # X = dataset.values[:, :-1].astype(float)
        # y = dataset.values[:, -1].astype(int)
        # y -= 1

        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        # test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        # return train_dataset, test_dataset
        dataset = openml.datasets.get_dataset('ovarianTumour')
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        y = X["class"].values.astype(np.int32)
        X = X.drop("class", axis=1).values.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    
    elif dataset_name=="Christine":
        dataset = openml.datasets.get_dataset('christine')
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        y = X["class"].values.astype(np.int32)
        X = X.drop("class", axis=1).values.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name=="Fabert":
        dataset = openml.datasets.get_dataset('fabert')
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        y = X["class"].values.astype(np.int32)
        X = X.drop("class", axis=1).values.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name=="Nomao":
        dataset = openml.datasets.get_dataset('nomao')
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        y = X["Class"].values.astype(np.int32)
        X = X.drop("Class", axis=1).values.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name=="Volkert":
        dataset = openml.datasets.get_dataset('volkert')
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        y = X["class"].values.astype(np.int32)
        X = X.drop("class", axis=1).values.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Connectionist":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data", header=None)
        X = dataset.values[:, :-1].astype(float)
        y = dataset.values[:, -1]
        y = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Dermatology":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data", header=None, na_values="?").dropna()
        X = dataset.values[:, :-1].astype(float)
        y = dataset.values[:, -1].astype(int) - 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Glass":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data", header=None, na_values="?").dropna()
        X = dataset.values[:, :-1].astype(float)
        y = dataset.values[:, -1].astype(int)
        y = LabelEncoder().fit_transform(y).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "CNAE9":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data", header=None, na_values="?").dropna()
        X = dataset.values[:, 1:].astype(float)
        y = dataset.values[:, 0].astype(int)
        y = LabelEncoder().fit_transform(y).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "ZOO":
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data")
        X = dataset.values[:, 1:-1].astype(float)
        y = dataset.values[:, -1]
        y = LabelEncoder().fit_transform(y).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Adult":
        dataset = datasets.load_dataset("scikit-learn/adult-census-income")
        df = pd.DataFrame(dataset['train'])

        y = df.pop('income').values
        y = LabelEncoder().fit_transform(y).astype(int)

        to_encode = ["workclass", "education", "marital.status", "occupation",
            "relationship", "race", "sex", "native.country"]
        for column in to_encode:
            df[column] = LabelEncoder().fit_transform(df[column])
           
        X = df.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test))

        return train_dataset, test_dataset
    elif dataset_name == "Helena":
        dataset = datasets.load_dataset("wwydmanski/helena", data_files="train_X.csv")
        df = pd.DataFrame(dataset['train'])
        X = df.values.astype(float)

        dataset = datasets.load_dataset("wwydmanski/helena", data_files="train_y.csv")
        df = pd.DataFrame(dataset['train'])
        y = df.values.astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dataset = data.TensorDataset(torch.tensor(X_train).type(torch.FloatTensor), torch.tensor(y_train).type(torch.FloatTensor))
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test).type(torch.FloatTensor))

        return train_dataset, test_dataset
    elif dataset_name == "Parkinsons":
        url = "https://archive.ics.uci.edu/static/public/174/parkinsons.zip"
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            r = requests.get(url)
            tmp.write(r.content)

        with open(tmp.name, "rb") as tmp:
            with zipfile.ZipFile(tmp, 'r') as zip_ref:
                df = zip_ref.read('parkinsons.data')

        df = pd.read_csv(StringIO(df.decode()))
        y = df["status"]
        X = df.drop(["status", "name"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y_train.values).type(torch.FloatTensor)

        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test.values).type(torch.FloatTensor))

        return train_dataset, test_dataset
    elif dataset_name == "Haberman":
        haberman_s_survival = fetch_ucirepo(id=43) 
        
        # data (as pandas dataframes) 
        X = haberman_s_survival.data.features 
        y = haberman_s_survival.data.targets 
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        labelEncoder = LabelEncoder()
        y_train = labelEncoder.fit_transform(y_train)
        y_test = labelEncoder.transform(y_test)

        X_train_tensor = torch.tensor(X_train).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y_train).type(torch.FloatTensor)

        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test).type(torch.FloatTensor))

        return train_dataset, test_dataset
    elif dataset_name == "Vertebral":
        haberman_s_survival = fetch_ucirepo(id=212) 
        
        # data (as pandas dataframes) 
        X = haberman_s_survival.data.features 
        y = haberman_s_survival.data.targets 
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        labelEncoder = LabelEncoder()
        y_train = labelEncoder.fit_transform(y_train)
        y_test = labelEncoder.transform(y_test)

        X_train_tensor = torch.tensor(X_train).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y_train).type(torch.FloatTensor)

        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test).type(torch.FloatTensor))

        return train_dataset, test_dataset
    elif dataset_name == "Ecoli":
        haberman_s_survival = fetch_ucirepo(id=39) 
        
        # data (as pandas dataframes) 
        X = haberman_s_survival.data.features 
        y = haberman_s_survival.data.targets 
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        labelEncoder = LabelEncoder()
        y_train = labelEncoder.fit_transform(y_train)
        y_test = labelEncoder.transform(y_test)

        X_train_tensor = torch.tensor(X_train).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y_train).type(torch.FloatTensor)

        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test).type(torch.FloatTensor))

        return train_dataset, test_dataset
    elif dataset_name == "Voting":
        haberman_s_survival = fetch_ucirepo(id=105) 
        
        # data (as pandas dataframes) 
        X = haberman_s_survival.data.features 
        y = haberman_s_survival.data.targets 

        le = LabelEncoder()
        X = X.apply(le.fit_transform)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        labelEncoder = LabelEncoder()
        y_train = labelEncoder.fit_transform(y_train)
        y_test = labelEncoder.transform(y_test)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train).type(torch.FloatTensor)
        y_train_tensor = torch.tensor(y_train).type(torch.FloatTensor)

        train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = data.TensorDataset(torch.tensor(X_test).type(torch.FloatTensor), torch.tensor(y_test).type(torch.FloatTensor))

        return train_dataset, test_dataset
    else:
        raise NotImplementedError
