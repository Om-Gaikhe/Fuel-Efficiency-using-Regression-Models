# Fuel-Efficiency-using-Regression-Models
This notebook focuses on predicting the fuel efficiency using Regression models. 

# Car Fuel Efficiency Prediction

This project is designed to compare traditional Machine Learning (ML) and Deep Learning (DL) methods for predicting the fuel efficiency of cars, using a dataset which contains features such as horsepower, engine size, curb weight, and number of cylinders. The fuel efficiency (city-MPG) of the car is predicted using these features.

## Features
1. Horsepower
2. Engine Size
3. Curb Weight
4. Number of Cylinders

## Target
1. City-MPG

## Procedures

This project is divided into four parts:

### 1. Data Preprocessing

The 'num-of-cylinders' column is converted from string format to integer format (e.g., 'four' is replaced with 4) using a Python dictionary. The features are then scaled using the Standard Scaler from the Scikit-Learn library.

### 2. Linear Regression 

The data is split into a 60-40 ratio for the train and test set respectively. A Linear Regression model from Scikit-Learn is then trained on the train set. The performance of the model is evaluated by predicting the fuel efficiency on the test set and calculating the Mean Absolute Error (MAE). The actual and predicted values are visualized using a scatter plot. 

### 3. Deep Learning 

A Neural Network model using TensorFlow is created with two hidden layers containing 64 and 32 neurons respectively. The model is trained for 100 epochs with a batch size of 100. A checkpoint is also set up to save the best performing model. The performance of the model is evaluated just as with the Linear Regression model. The loss is plotted against the number of epochs to visualize the training and validation loss.

### 4. Performance Comparison

The performances of the Linear Regression and Neural Network models are compared by plotting the Mean Absolute Error (MAE) of both models. In this case, the Linear Regression model performed better, with a MAE of 2.34, compared to the Neural Network model, with a MAE of 3.66.

## Insights

1. From the pairplot, we observe that the features follow a normal distribution which is a good sign for applying linear regression.
2. The scatter plot of predicted vs actual values for the linear regression model suggests a good fit for the data, indicating that the model was able to learn the underlying relationship between the features and target variable.
3. From the Neural Network model's loss plot, we can observe that the validation loss slightly increases after a certain number of epochs, indicating overfitting.
4. The comparison of MAE shows that the linear regression model outperforms the neural network model in this case. This could be due to the relatively small size of the dataset, which might not be enough for the neural network to fully learn and generalize.

## Future Work

1. Different configurations of the Neural Network can be tested to see if performance improves.
2. Other ML models, like Decision Trees or Random Forests, can be used and their performance compared.
3. More feature engineering can be applied to improve the models' performance. 

**Note:** This is a learning project aimed at comparing the performance of traditional machine learning models and deep learning models on a relatively small dataset. The dataset used is a cleaned version of an open source dataset and may not reflect the real-world complexities that larger and more varied datasets might have.
