#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for the admissions prediction model.

This script:
1. Loads the raw admission.csv data
2. Cleans and preprocesses the data
3. Splits into training and testing sets
4. Optionally applies feature scaling
5. Saves the processed datasets to the processed data directory
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: The loaded dataframe
    """
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Clean and preprocess the data.
    
    Args:
        df (pandas.DataFrame): Raw dataframe
        
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    print("Preprocessing data...")
    
    # Make a copy to avoid modifying the original dataframe
    df_cleaned = df.copy()
    
    # Remove the Serial No. column if it exists
    if 'Serial No.' in df_cleaned.columns:
        df_cleaned.drop('Serial No.', axis=1, inplace=True)
        print("Removed 'Serial No.' column")
    
    # Check for missing values
    missing_values = df_cleaned.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values")
        # For this example, we'll drop rows with missing values
        # In a real scenario, you might want to impute them
        df_cleaned.dropna(inplace=True)
        print(f"Dropped rows with missing values, {len(df_cleaned)} rows remaining")
    else:
        print("No missing values found")
    
    return df_cleaned

def split_and_scale_data(df, target_column='Chance of Admit ', test_size=0.2, random_state=42, apply_scaling=True):
    """
    Split data into training and testing sets, and optionally apply feature scaling.
    
    Args:
        df (pandas.DataFrame): Cleaned dataframe
        target_column (str): Name of the target variable column
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        apply_scaling (bool): Whether to apply feature scaling
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print(f"Splitting data with test_size={test_size} and random_state={random_state}")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Apply feature scaling if specified
    scaler = None
    if apply_scaling:
        print("Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler

def save_datasets(X_train, X_test, y_train, y_test, output_dir):
    """
    Save the processed datasets to CSV files.
    
    Args:
        X_train, X_test, y_train, y_test: Processed datasets
        output_dir (str): Directory to save the processed files
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"Saved processed datasets to {output_dir}")

def main():
    """Main function to execute the data preparation pipeline."""
    # Define paths
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(current_dir, 'data', 'raw', 'admission.csv')
    processed_data_path = os.path.join(current_dir, 'data', 'processed')
    
    # Load data
    df = load_data(raw_data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Preprocess data
    df_cleaned = preprocess_data(df)
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(df_cleaned)
    
    # Save processed datasets
    save_datasets(X_train, X_test, y_train, y_test, processed_data_path)
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()