# titanic_assignment
Titanic Dataset Analysis 
# Titanic Dataset Analysis - Predictive Modeling Assignment

## 📋 Project Overview
This project performs comprehensive analysis and feature engineering on the Titanic dataset to build a predictive model for passenger survival. The project follows best practices in data cleaning, feature engineering, and feature selection.

## 🏗️ Project Structure
titanic_assignment/
│
├── data/ # Dataset files
│ ├── train.csv # Original training data
│ ├── test.csv # Original test data
│ ├── train_cleaned.csv # After cleaning
│ ├── test_cleaned.csv # After cleaning
│ ├── train_engineered.csv # After feature engineering
│ ├── test_engineered.csv # After feature engineering
│ └── selected_features.txt # Final feature list
│
├── notebooks/ # Jupyter notebooks
│ └── Titanic_Feature_Engineering.ipynb # Main exploration notebook
│
├── scripts/ # Python scripts
│ ├── data_cleaning.py # Missing values, outliers
│ ├── feature_engineering.py # Create new features
│ └── feature_selection.py # Correlation, importance analysis
│
├── README.md # This file
└── requirements.txt # Dependencies


# Step 1: Data Cleaning
python scripts/data_cleaning.py

# Step 2: Feature Engineering
python scripts/feature_engineering.py

# Step 3: Feature Selection
python scripts/feature_selection.py