import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split


#Pytorch dataset class for MetroPT dataset
class MetroPTDataset(Dataset):
    @staticmethod
    def __tabular_to_window(x,y,window_size):
        '''
        Transform tabular data to windowed data in sliding window form with window size as a parameter.
        x: features
        y: target
        window_size: size of the window in number of observations
        '''
        x = np.array(x)
        y = np.array(y)
        #print(y)
        features = []
        labels = []
        for i in range(window_size,len(x)):
            features.append(x[i-window_size:i])
            labels.append(y[i])
        return np.array(features), np.array(labels)
    @staticmethod
    def __dataset_from_subsets(subsets, data_mean, data_std, target,window_size,drop_cols=[],scaler=None,feature_selection=None):
        '''
        Pick a bunch of datasets, perform window tranformation and join all the windowed
        data on a unique dataset
        subset: iterable containing the subsets to join
        target: target variable labels in the subsets
        window_size: size of the window to group the data
        drop_cols: list of column labels that have to be remove form the final dataset
        '''
        features = []
        labels = []
        for subset in subsets:
            if scaler != None:
                x_subset = scaler.transform(subset.drop(columns=[target,*drop_cols]))
            else:
                x_subset = subset.drop(columns=[target,*drop_cols])
                x_subset = (x_subset - data_mean) / data_std
            if len(subset) > window_size:
                x, y = MetroPTDataset.__tabular_to_window(x_subset,subset[target],window_size=window_size)
                features.append(x)#np.concatenate((features,x))
                labels.append(y)#np.concatenate((labels,y))
                print(len(features[-1]))
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        #features = features.reshape(-1, features.shape[-1])
        #labels = labels.reshape(-1, labels.shape[-1])
        #print(features.shape, labels.shape)
        return features, labels

    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
    
    @staticmethod
    def __get_subsets(data_dir,piece_wise_rul = 0):
        # metro_pt1 = pd.read_csv(data_dir+'/MetroPT.csv')
        metro_pt2 = pd.read_csv(data_dir+'/MetroPT2.csv')
        metro_pt3 = pd.read_csv(data_dir+'/MetroPT3.csv')

        # To datetime conversion
        # metro_pt1['timestamp'] = pd.to_datetime(metro_pt1['timestamp'])
        metro_pt2['timestamp'] = pd.to_datetime(metro_pt2['timestamp'])
        metro_pt3['timestamp'] = pd.to_datetime(metro_pt3['timestamp'])

        # Drop gps data
        # metro_pt1.drop(columns=['gpsLat', 'gpsLong', 'gpsSpeed', 'gpsQuality','Flowmeter'], inplace=True)
        metro_pt2.drop(columns=['gpsLat', 'gpsLong', 'gpsSpeed', 'gpsQuality','Flowmeter','COMP'], inplace=True)
        metro_pt3.drop(columns=['Unnamed: 0','COMP'], inplace=True)

        metro_pt2.ffill(inplace=True)
        metro_pt3.ffill(inplace=True)
        metro_pt2.dropna(inplace=True)
        metro_pt3.dropna(inplace=True)

        # metro_pt1['rul'] = 0.0
        metro_pt2['rul'] = 0.0
        metro_pt3['rul'] = 0.0
        # create run-to-failure experiment subsets and compute RUL
        subsets = []

        # inicio_run = [0, 4247004, 5707697, 10739274]
        # fallo = [4232184,5705897, 10543791]
        # for i in range(0,len(fallo)):
        #     ultima_fecha = metro_pt1['timestamp'][fallo[i]]
        #     (metro_pt1['rul'][inicio_run[i]:fallo[i]]) = (metro_pt1['timestamp'][inicio_run[i]:fallo[i]]).apply(lambda x: (ultima_fecha - x).total_seconds() /(60*60) )
        #     subsets.append(metro_pt1[inicio_run[i]:fallo[i]])

        inicio_run = [0, 2949092, 5998011]
        fallo = [2934365,5779176]
        for i in range(0,len(fallo)):
            ultima_fecha = metro_pt2['timestamp'][fallo[i]]
            (metro_pt2['rul'][inicio_run[i]:fallo[i]]) = (metro_pt2['timestamp'][inicio_run[i]:fallo[i]]).apply(lambda x: (ultima_fecha - x).total_seconds() /(60*60*24) )
            subsets.append(metro_pt2[inicio_run[i]:fallo[i]])

        # Remove the last run from the dataset
        subsets = subsets[:-1]

        inicio_run = [0, 571226, 843101, 908124, 1172714]
        fallo = [562564,840740, 890810, 1171093]
        for i in range(0,len(fallo)):
            ultima_fecha = metro_pt3['timestamp'][fallo[i]]
            (metro_pt3['rul'][inicio_run[i]:fallo[i]]) = (metro_pt3['timestamp'][inicio_run[i]:fallo[i]]).apply(lambda x: (ultima_fecha - x).total_seconds() /(60*60*24) )
            subsets.append(metro_pt3[inicio_run[i]:fallo[i]])
        
        # downsample subsets to 1min frequency
        for i in range(len(subsets)):
            metropt_hours = subsets[i].drop(columns=['rul']).groupby(pd.Grouper(key='timestamp', freq='1min',dropna=True)).mean()
            rul_hours = subsets[i][['timestamp','rul']].groupby(pd.Grouper(key='timestamp', freq='1min',dropna=True)).min()
            # metropt_hours = subsets[i].drop(columns=['rul']).groupby(pd.Grouper(key='timestamp', freq='1min',dropna=True)).mean()
            # rul_hours = subsets[i][['timestamp','rul']].groupby(pd.Grouper(key='timestamp', freq='1min',dropna=True)).min()
            subsets[i] = pd.merge(metropt_hours, rul_hours, on='timestamp')
            subsets[i].dropna(inplace=True)
        
        # piece-wise rul transformation
        if piece_wise_rul > 0:
            for i in range(len(subsets)):
                subsets[i]['rul'] = subsets[i]['rul'].apply(lambda x: np.min([piece_wise_rul,x]))
        return subsets

    @staticmethod        
    def get_datasets(data_dir, test_pct=0.7, piece_wise_rul = 0, window_size = 30, validation_rate=0):
        '''
        Load the MetroPT dataset and split it into train and test datasets
        data_dir: directory where the dataset is stored
        test_pct: percentage of the data to be used as test data
        piece_wise_rul: maximum RUL value for each experiment
        window_size: size of the window to group the data

        returns: train_dataset, test_dataset
        '''
        subsets = MetroPTDataset.__get_subsets(data_dir)

        # add experiment number to each subset
        # for i in range(len(subsets)):
        #     subsets[i]['experiment'] = i+1

        # separate train and test data
        n_subsets_train = int(test_pct*len(subsets))
        train_subsets = subsets[:n_subsets_train]
        test_subsets = subsets[n_subsets_train:]

        # Transformations

        # Normalize data
        data_mean, data_std = MetroPTDataset.get_normalization_parameters(train_subsets)
        # for subset in train_subsets:
        #     subset = (subset - data_mean) / data_std
        # for subset in test_subsets:
        #     subset = (subset - data_mean) / data_std


        # piece-wise rul transformation
        if piece_wise_rul > 0:
            for i in range(len(subsets)):
                subsets[i]['rul'] = subsets[i]['rul'].apply(lambda x: np.min([piece_wise_rul,x]))

        # sliding window transformation
        if window_size > 1:
            x_train, y_train = MetroPTDataset.__dataset_from_subsets(train_subsets, data_mean, data_std, target='rul',window_size=window_size)
            x_test, y_test = MetroPTDataset.__dataset_from_subsets(test_subsets, data_mean, data_std, target='rul',window_size=window_size)
        else:
            # concatenate all subsets
            train_data = pd.concat(train_subsets)
            test_data = pd.concat(test_subsets)
            x_train = train_data.drop(columns=['rul'])
            y_train = train_data['rul']
            x_test = test_data.drop(columns=['rul'])
            y_test = test_data['rul']
        
        if validation_rate > 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_rate, random_state=42)
            train_dataset = MetroPTDataset(x_train, y_train)
            val_dataset = MetroPTDataset(x_val, y_val)
        else:
            train_dataset = MetroPTDataset(x_train, y_train)
            val_dataset = None
        test_dataset = MetroPTDataset(x_test, y_test)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def get_dataloaders(data_dir, test_pct=0.7, piece_wise_rul = 0, window_size = 30, batch_size=32,validation_rate=0):
        '''
        Load the MetroPT dataset and split it into train and test datasets
        data_dir: directory where the dataset is stored
        test_pct: percentage of the data to be used as test data
        piece_wise_rul: maximum RUL value for each experiment
        window_size: size of the window to group the data
        batch_size: size of the batch for the dataloaders

        returns: train_dataloader, test_dataloader
        '''
        train_dataset, val_dataset, test_dataset = MetroPTDataset.get_datasets(data_dir, test_pct, piece_wise_rul, window_size)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if validation_rate > 0:
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else: 
            val_dataloader = None
        return train_dataloader, val_dataloader, test_dataloader
    
    @staticmethod
    def get_normalization_parameters(train_subsets):
        '''
        Compute the mean and standard deviation of the training data to normalize the data
        train_subsets: list of subsets to compute the normalization parameters
        '''
        data = pd.concat(train_subsets)
        data_mean = data.drop(columns=['rul']).mean()
        data_std = data.drop(columns=['rul']).std()

        return data_mean, data_std
    @staticmethod
    def get_leave_one_out_dataloaders(data_dir, piece_wise_rul = 0, window_size = 30, batch_size=32):
        splits = []
        subsets = MetroPTDataset.__get_subsets(data_dir,piece_wise_rul)

        for i in range(len(subsets)):
            train_subsets = subsets[:i] + subsets[i+1:]
            test_subsets = [subsets[i]]

            # Normalize data
            data_mean, data_std = MetroPTDataset.get_normalization_parameters(train_subsets)

            x_train, y_train = MetroPTDataset.__dataset_from_subsets(train_subsets, data_mean, data_std, target='rul',window_size=window_size)
            x_test, y_test = MetroPTDataset.__dataset_from_subsets(test_subsets, data_mean, data_std, target='rul',window_size=window_size)
            # 80/20 split from training for validation
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

            train_dataset = MetroPTDataset(x_train, y_train)
            val_dataset = MetroPTDataset(x_val, y_val)
            test_dataset = MetroPTDataset(x_test, y_test)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12,persistent_workers=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            splits.append((train_dataloader, val_dataloader, test_dataloader))

        return splits

    @staticmethod
    def get_leave_one_out_dataloaders_by_index(data_dir, piece_wise_rul = 0, window_size = 30, batch_size=32,validation_rate=0,index=0):
        print('Getting leave one out dataloaders')
        splits = []
        subsets = MetroPTDataset.__get_subsets(data_dir,piece_wise_rul)
        print(f'{len(subsets)} subsets loaded')
        print(f'Using subset {index} as test data')
        # remover subset smaller than window size
        subsets = [subset for subset in subsets if len(subset) > window_size]

        train_subsets = subsets[:index] + subsets[index+1:]
        test_subsets = [subsets[index]]
        # Normalize data
        data_mean, data_std = MetroPTDataset.get_normalization_parameters(train_subsets)

        x_train, y_train = MetroPTDataset.__dataset_from_subsets(train_subsets, data_mean, data_std, target='rul',window_size=window_size)
        x_test, y_test = MetroPTDataset.__dataset_from_subsets(test_subsets, data_mean, data_std, target='rul',window_size=window_size)
        # 80/20 split from training for validation
        if validation_rate == 0:
            val_dataloader = None
        else:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_rate, shuffle=False)#random_state=42)
            val_dataset = MetroPTDataset(x_val, y_val)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        train_dataset = MetroPTDataset(x_train, y_train)
        test_dataset = MetroPTDataset(x_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)#, persistent_workers=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_dataloader, val_dataloader, test_dataloader


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor([self.labels[idx]])

        if self.transform:
            sample = self.transform(sample)

        return sample, label