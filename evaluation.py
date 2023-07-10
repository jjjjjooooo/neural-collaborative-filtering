import os
import torch
import numpy as np
from tqdm import tqdm

import data as d
import model as m

# Get the current working directory
cwd = os.getcwd()

# Split the ratings data
df, df_train, df_test = d.split_ratings_data(
    os.path.join(cwd, 'ml-25m', 'ratings.csv'))

# Determine the number of users and items
num_users = df['userId'].max() + 1
num_items = df['movieId'].max() + 1

# Create the model
model = m.NCF(num_users, num_items)

# Load the saved model state dict
model.load_state_dict(torch.load('best_model.pt'))
# trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

# User-item pairs for testing
user_item_pair_test = set(zip(df_test['userId'], df_test['movieId']))

# Dict of all items that are interacted with by each user
user_items_dict = df.groupby('userId')['movieId'].apply(list).to_dict()

# List of all movie ids
all_movie_ids = df['movieId'].unique()

# Set the model to evaluation mode
model.eval()

hits_top5 = []
hits_top10 = []
for (u, i) in tqdm(user_item_pair_test):
    interacted_items = user_items_dict[u]
    not_interacted_items = set(all_movie_ids) - set(interacted_items)
    selected_not_interacted_items = list(
        np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted_items + [i]

    user_input = torch.tensor([u] * 100)
    item_input = torch.tensor(test_items)

    # Disable gradient computation during inference
    with torch.no_grad():
        predicted_labels = model(user_input,
                                 item_input).detach().squeeze().numpy()

    top5_items = [
        test_items[i] for i in np.argsort(predicted_labels)[::-1][:5]
    ]

    top10_items = [
        test_items[i] for i in np.argsort(predicted_labels)[::-1][:10]
    ]

    if i in top5_items:
        hits_top5.append(1)
    else:
        hits_top5.append(0)

    if i in top10_items:
        hits_top10.append(1)
    else:
        hits_top10.append(0)

# Calculate and print the Hit Ratio @ 5
hit_top5_ratio = np.average(hits_top5) * 100
print("Hit Ratio @ 5: {:.2f}%".format(hit_top5_ratio))

# Calculate and print the Hit Ratio @ 10
hit_top10_ratio = np.average(hits_top10) * 100
print("Hit Ratio @ 10: {:.2f}%".format(hit_top10_ratio))

# Set the model to train mode
model.train()