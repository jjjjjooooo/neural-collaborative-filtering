# Neural Collaborative Filtering 

## Goals
Building a recommendation systems with neural networks to predict whether an user will interact with each movie, with the aim of presenting to users the movies with the highest interaction likelihood.

## Datasets
MovieLens 25M Dataset <br />
<https://grouplens.org/datasets/movielens/> <br />

The dataset used in this project is ratings.csv, which originally contains 25000095 entries based on explicit feedback. The recommender system is trained using implicit feedback by binarizing the ratings.

## Files
* eda.ipynb: Exploratory data analysis to gather an overview on the dataset. <br />
* data.py: It contains functions and classes to prepare the data on top of pytorch lightning for the neural network then. <br />
* model.py: Neural collaborative filtering based on pytorch lightning. <br />
* trainer.py: All setups related to the training process. <br />
* evaluation.ipynb: Evaluating the perfomance of the trained model with respect to the metrices of Hit Ratio @ 5 and Hit Ratio @ 10. <br />
* best_model.pt: The trained model with the lowest val_loss. <br />

## Performance
Hit Ratio @ 5: 90.41% <br />
Hit Ratio @ 10: 96.49% <br />
