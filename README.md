# Neural Collaborative Filtering 

## Goals
Building a recommendation systems with neural networks to predict whether an user will interact with each movie, with the goal of recommending the movies to users with the highest interaction likelihood.

## Datasets
MovieLens 25M Dataset <br />
<https://grouplens.org/datasets/movielens/> <br />

The dataset employed in this project is **ratings.csv**, comprising 25,000,095 entries derived from explicit feedback. The recommender system is trained using implicit feedback by binarizing the explicit ratings, resulting in an expanded dataset of approximately 125,000,000 entries, making it five times larger.

## Files
* **eda.ipynb**: Exploratory data analysis on the dataset. <br />
* **data.py**: Functions and classes for preparing the data based on pytorch lightning for the neural network then. <br />
* **model.py**: Architecture of the neural collaborative filtering based on pytorch lightning for training. <br />
* **trainer.py**: Main file to execute the training process. <br />
* **best_model.pt**: The trained model hitting the lowest val_loss. <br />
* **evaluation.ipynb**: File for evaluating the trained model with the metrices being Hit Ratio @ 5 and Hit Ratio @ 10. <br />


## Performances
Hit Ratio @ 5: 90.41% <br />
Hit Ratio @ 10: 96.49% <br />
