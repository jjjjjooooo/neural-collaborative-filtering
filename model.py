import torch
import torch.nn as nn
import pytorch_lightning as pl


class NCF(pl.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim=8):
        """
        Neural Collaborative Filtering (NCF) model for recommendation.

        Args:
            num_users (int): The number of unique users.
            num_items (int): The number of unique items.
            embedding_dim (int): The dimensionality of user and item embeddings. Default is 8.
        """
        super().__init__()

        self.save_hyperparameters()

        self.user_embedding = nn.Embedding(num_embeddings=num_users,
                                           embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items,
                                           embedding_dim=embedding_dim)
        self.fc1 = nn.Linear(in_features=embedding_dim * 2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_input, item_input):
        """
        Forward pass of the NCF model.

        Args:
            user_input (torch.Tensor): Tensor of user IDs.
            item_input (torch.Tensor): Tensor of item IDs.

        Returns:
            torch.Tensor: Predicted ratings.
        """
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        vector = nn.ReLU()(self.fc1(vector))
        vector = self.dropout1(vector)
        vector = nn.ReLU()(self.fc2(vector))
        vector = self.dropout2(vector)
        pred = self.output(vector)
        return pred

    def training_step(self, batch, batch_idx):
        """
        Training step of the NCF model.

        Args:
            batch (tuple): Tuple containing user_input, item_input, and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        user_input, item_input, labels = batch
        predicted_labels = self.forward(user_input, item_input)
        loss = nn.BCEWithLogitsLoss()(predicted_labels, labels.view(-1,
                                                                    1).float())
        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the NCF model.

        Args:
            batch (tuple): Tuple containing user_input, item_input, and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        user_input, item_input, labels = batch
        predicted_labels = self.forward(user_input, item_input)
        loss = nn.BCEWithLogitsLoss()(predicted_labels, labels.view(-1,
                                                                    1).float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for the NCF model.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer