# Neural Collaborative Filtering 

## Goals
Building a recommendation systems with neural networks to predict whether an user will interact with each movie, with the goal of recommending the movies to users with the highest interaction likelihood.

## Datasets
MovieLens 25M Dataset <br />
<https://grouplens.org/datasets/movielens/> <br />

The dataset employed in this project is **ratings.csv**, comprising 25,000,095 entries derived from explicit feedback. The recommender system is trained using implicit feedback by binarizing the explicit ratings, resulting in an expanded dataset of approximately 125,000,000 entries, making it five times larger.

## Files
* **eda.ipynb**: This notebook contains the code for conducting exploratory data analysis on the dataset.
* **data.py**: This file includes functions and classes responsible for preparing the data to be used with the neural network, utilizing pytorch lightning.
* **model.py**: The code in this file outlines the architecture of the neural collaborative filtering, which is implemented using pytorch lightning for the training process.
* **trainer.py**: This is the main file to execute and oversee the training process.
* **best_model.pt**: The trained model with the lowest validation loss is saved in this file.
* **evaluation.ipynb**: In this notebook, the trained model is evaluated using the metrics Hit Ratio @ 5 and Hit Ratio @ 10.

## Architecture of Neural Network
In this project, label encoding is utilized for both the input variables of userId and movieId, as well as the output variable indicating whether an interaction has occurred (labeled as 1) or not (labeled as 0).

![image](https://github.com/jjjjjooooo/neural-collaborative-filtering/assets/50882720/d775e74a-7001-41ad-9f1e-23f8e1284e4b)


## Performances
Hit Ratio @ 5: 90.41% <br />
Hit Ratio @ 10: 96.49% <br />

