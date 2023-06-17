import data as d
import model as m
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import wandb
import time

# Set the working directory
os.chdir(r'D:\Dropbox\Self-Development\Coding_Projects\RecommenderSystem')

# Login to Wandb
wandb.login()

# Initialize Wandb run
wandb.init()

# Configure Wandb logger
logger = WandbLogger(
    name="ncf",
    entity='jo',
    log_model=True,
)

# Get the current working directory
cwd = os.getcwd()

# Configure EarlyStopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode='min')

# Configure ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='sample-movielens-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min')

# Split the ratings data
df, df_train, df_test = d.split_ratings_data(
    os.path.join(cwd, 'ml-25m', 'ratings.csv'))

# Create the dataset
dataset = d.MovieLensTrainDataset(df, df_train)

# Create the data module
dm = d.MovieLensTrainDataModule(dataset)

# Determine the number of users and items
num_users = df['userId'].max() + 1
num_items = df['movieId'].max() + 1

# Create the model
model = m.NCF(num_users, num_items)

# Create the trainer
trainer = pl.Trainer(max_epochs=1,
                     min_epochs=1,
                     profiler='simple',
                     accelerator="gpu",
                     devices=-1,
                     precision=16,
                     callbacks=[early_stopping, checkpoint_callback],
                     val_check_interval=0.1,
                     logger=logger,
                     accumulate_grad_batches=1)

# Train the model
trainer.fit(model, datamodule=dm)
#trainer.save_checkpoint("checkpoint.ckpt")

# Save the best model checkpoint
checkpoint_callback.best_model_path

model_name = f'best_model-{int(time.time())}'
torch.save(model.state_dict(), model_name)

# Finish Wandb run
wandb.finish()