# **Crime Prediction and Analysis using Machine Learning**

This project leverages machine learning techniques to predict and analyze crime trends using historical police crime records. It employs multiple machine learning models, including Random Forest and Gradient Boosting classifiers, to provide insights that can help reduce crime prediction errors.

## **Overview**
The project aims to build a robust crime prediction system by analyzing time-series and categorical features such as year, month, day, hour, and location attributes. The solution provides performance evaluation metrics for each model and suggests optimal techniques for enhancing prediction accuracy.

## **Dataset**
The dataset used contains crime reports from 2014 to 2017, with features including:
- **Temporal Information:** Year, month, day, hour, etc.
- **Geographical Information:** Division, neighborhood ID (HoodId)
- **Crime Details:** Crime type (CDA), and premise type.

## **Features**
The following features are used for crime prediction:
- **Temporal Attributes:** Year, month, day, and hour of crime occurrences.
- **Geographical Attributes:** Division and neighborhood information.
- **Categorical Encoding:** Premise type and crime categories.

## **Models**
The project implements and evaluates the following models:
- **Random Forest Classifier:** A robust ensemble model for classification tasks.
- **One-Hot Encoded Random Forest Classifier:** Enhanced version using encoded categorical data.
- **Balanced Random Forest Classifier:** Optimized for handling imbalanced datasets.
- **Gradient Boosting Classifier:** A boosting method for classification tasks.

## **Results**
| **Model**                       | **Accuracy** |
|----------------------------------|--------------|
| Random Forest                    | 0.6474       |
| One-Hot Encoded Random Forest     | 0.6616       |
| Balanced Random Forest            | 0.6454       |
| Gradient Boosting                 | 0.5588       |
