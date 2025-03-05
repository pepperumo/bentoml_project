#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training script for the admissions prediction model.

This script:
1. Loads the processed datasets (X_train, X_test, y_train, y_test)
2. Trains a regression model to predict admission chances
3. Evaluates the model performance using metrics
4. Saves the trained model to the BentoML model store
"""

import os
import pandas as pd
import numpy as np
import bentoml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

def load_processed_data(processed_data_dir):
    """
    Load the processed train and test datasets.
    
    Args:
        processed_data_dir (str): Directory containing processed datasets
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Loading processed datasets from {processed_data_dir}...")
    
    # Load datasets
    X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'))
    
    # Convert y_train and y_test from DataFrames to Series
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    
    print(f"Loaded training set: {X_train.shape}, {y_train.shape}")
    print(f"Loaded testing set: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a regression model.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        object: Trained model
    """
    print("Training LinearRegression model...")
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model coefficients
    print("Model coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model (object): Trained model
        X_test (pandas.DataFrame): Testing features
        y_test (pandas.Series): Testing target
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    print("Evaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    }

def save_model_with_bentoml(model, model_name="admissions_model"):
    """
    Save the trained model to BentoML model store.
    
    Args:
        model (object): Trained model
        model_name (str): Name to give the model in the BentoML store
        
    Returns:
        str: Path to the saved model in the BentoML store
    """
    print(f"Saving model to BentoML model store as '{model_name}'...")
    
    # Save model
    saved_model = bentoml.sklearn.save_model(
        model_name,
        model,
        signatures={
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
        metadata={
            "description": "Linear regression model for predicting admission chances",
        }
    )
    
    print(f"Model saved successfully. Tag: {saved_model.tag}")
    print(f"Use 'bentoml models get {saved_model.tag}' to view model details")
    
    return saved_model.path

def main():
    """Main function to execute the model training pipeline."""
    # Define paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(current_dir, 'data', 'processed')
    
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data(processed_data_path)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # If performance is acceptable, save the model
    if metrics['r2'] > 0.7:  # Example threshold
        print("Model performance is acceptable, saving to BentoML model store...")
        save_model_with_bentoml(model)
    else:
        print("Model performance is below threshold, consider retraining with different parameters.")
    
    print("Model training completed!")

if __name__ == "__main__":
    main()