# -*- coding: utf-8 -*-
"""
Data Preparation for MovieLens Collaborative Filtering

Author: [Ou Jin]
"""

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset


def split_ratings_data(dir):
    """
    Split the ratings data into training and test sets.

    Args:
        dir (str): The file path of the ratings data.
        num_samples (int): The number of rows to sample from the ratings data. Default is 1,000,000.

    Returns:
        pandas.DataFrame: The original ratings data containing columns 'userId', 'movieId', and 'timestamp'.
        pandas.DataFrame: The training ratings data containing columns 'userId', 'movieId', and 'is_rated'.
        pandas.DataFrame: The test ratings data containing columns 'userId' and 'movieId'.
    """
    # Load the ratings data from the file and sample a specified number of rows
    df = pd.read_csv(dir, parse_dates=["timestamp"])

    # Calculate the rank based on timestamp for each user
    df["rank"] = df.groupby("userId")["timestamp"].rank(method="first", ascending=False)

    # Create the training ratings data by excluding rows with rank equal to 1
    df_train = df.loc[df["rank"] != 1, ["userId", "movieId"]]
    df_train["is_rated"] = 1

    # Create the test ratings data by selecting rows with rank equal to 1
    df_test = df.loc[df["rank"] == 1, ["userId", "movieId"]]

    return df, df_train, df_test


class MovieLensTrainDataset(Dataset):
    def __init__(self, df, df_train):
        """
        MovieLens training dataset for collaborative filtering.

        Args:
            df (pandas.DataFrame): The original ratings data containing columns 'userId', 'movieId'.
            df_train (pandas.DataFrame): The training ratings data containing columns 'userId', 'movieId'.
        """
        self.users, self.items, self.labels = self.get_dataset(df, df_train)

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.users)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            torch.Tensor: The user ID.
            torch.Tensor: The item ID.
            torch.Tensor: The label (1 for positive sample, 0 for negative sample).
        """
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, df, df_train):
        """
        Prepare the training dataset for collaborative filtering.

        Args:
            df (pandas.DataFrame): The original ratings data containing columns 'userId', 'movieId'.
            df_train (pandas.DataFrame): The training ratings data containing columns 'userId', 'movieId'.

        Returns:
            torch.Tensor: Tensor of user IDs.
            torch.Tensor: Tensor of item IDs.
            torch.Tensor: Tensor of labels.
        """
        movieId = df["movieId"].unique()

        user_item = set(zip(df["userId"], df["movieId"]))
        user_item_train = set(zip(df_train["userId"], df_train["movieId"]))

        users, items, labels = [], [], []
        num_negatives = 4
        for u, i in tqdm(user_item_train, desc="Creating training dataset"):
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(movieId)
                while (u, negative_item) in user_item:
                    negative_item = np.random.choice(movieId)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


class MovieLensTrainDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=1024):
        """
        MovieLens training data module for PyTorch Lightning.

        Args:
            dataset (torch.utils.data.Dataset): The training dataset.
            batch_size (int): The batch size. Default is 64.
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        """
        Split the dataset into training and validation sets.

        Args:
            stage: Not used in this implementation.
        """
        full_ds_size = len(self.dataset)
        train_ds_size = round(full_ds_size * 0.9)
        val_ds_size = full_ds_size - train_ds_size
        self.ds_train, self.ds_val = random_split(self.dataset, [train_ds_size, val_ds_size])

    def train_dataloader(self):
        """
        Create a DataLoader for the training set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the training set.
        """
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Create a DataLoader for the validation set.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation set.
        """
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False)
