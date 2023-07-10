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
* **best_model.pt**: The trained model hitting the lowest validation loss. <br />
* **evaluation.ipynb**: File for evaluating the trained model with the metrices being Hit Ratio @ 5 and Hit Ratio @ 10. <br />

## Architecture of Neural Network
In this project, label encoding is utilized for both the input variables of userId and movieId, as well as the output variable indicating whether an interaction has occurred (labeled as 1) or not (labeled as 0).

![image](https://github.com/jjjjjooooo/neural-collaborative-filtering/assets/50882720/d775e74a-7001-41ad-9f1e-23f8e1284e4b)


## Performances
Hit Ratio @ 5: 90.41% <br />
Hit Ratio @ 10: 96.49% <br />

