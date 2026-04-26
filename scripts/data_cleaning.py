"""
Titanic Dataset - Data Cleaning Module
This script handles missing values, outliers, and data consistency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use non-interactive backend to prevent blocking
plt.switch_backend('Agg')

def load_data(train_path, test_path):
    """Load the Titanic dataset"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Training data loaded: {train.shape}")
    print(f"Test data loaded: {test.shape}")
    return train, test

def analyze_missing_values(df, name):
    """Analyze and report missing values"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        print(f"\nMissing Values in {name}:")
        print(missing_df.to_string(index=False))
    else:
        print(f"\nNo missing values in {name}")
    
    return missing_df

def handle_missing_values(train, test):
    """Handle all missing values in the dataset"""
    print("\n" + "="*50)
    print("HANDLING MISSING VALUES")
    print("="*50)
    
    # Create copies
    train_clean = train.copy()
    test_clean = test.copy()
    
    # 1. Age - impute with median
    age_median = train['Age'].median()
    print(f"\n1. Age: Imputing {train['Age'].isnull().sum()} missing values with median: {age_median:.1f}")
    train_clean['Age'] = train_clean['Age'].fillna(age_median)
    test_clean['Age'] = test_clean['Age'].fillna(age_median)
    
    # Create missing indicator
    train_clean['Age_Missing'] = train['Age'].isnull().astype(int)
    test_clean['Age_Missing'] = test['Age'].isnull().astype(int)
    print("   - Added 'Age_Missing' indicator column")
    
    # 2. Embarked - fill with mode
    if train['Embarked'].isnull().sum() > 0:
        embarked_mode = train['Embarked'].mode()[0]
        print(f"\n2. Embarked: Filling {train['Embarked'].isnull().sum()} missing with mode: {embarked_mode}")
        train_clean['Embarked'] = train_clean['Embarked'].fillna(embarked_mode)
    
    # 3. Fare - fill with median in test
    if test['Fare'].isnull().sum() > 0:
        fare_median = train['Fare'].median()
        print(f"\n3. Fare: Filling {test['Fare'].isnull().sum()} missing in test with median: {fare_median:.2f}")
        test_clean['Fare'] = test_clean['Fare'].fillna(fare_median)
    
    # 4. Cabin - create indicator and drop
    cabin_missing_pct = (train['Cabin'].isnull().sum() / len(train)) * 100
    print(f"\n4. Cabin: {cabin_missing_pct:.1f}% missing - dropping column")
    train_clean['Cabin_Missing'] = train['Cabin'].isnull().astype(int)
    test_clean['Cabin_Missing'] = test['Cabin'].isnull().astype(int)
    train_clean = train_clean.drop('Cabin', axis=1)
    test_clean = test_clean.drop('Cabin', axis=1)
    print("   - Added 'Cabin_Missing' indicator, dropped original column")
    
    return train_clean, test_clean

def check_consistency(train_clean, test_clean):
    """Check and fix data consistency issues"""
    print("\n" + "="*50)
    print("CHECKING DATA CONSISTENCY")
    print("="*50)
    
    # Check Sex values
    print(f"\nSex values in train: {train_clean['Sex'].unique()}")
    print(f"Sex values in test: {test_clean['Sex'].unique()}")
    
    # Standardize if needed
    train_clean['Sex'] = train_clean['Sex'].map({'male': 'male', 'female': 'female'})
    test_clean['Sex'] = test_clean['Sex'].map({'male': 'male', 'female': 'female'})
    print("Sex values standardized")
    
    # Check duplicates
    train_dupes = train_clean.duplicated().sum()
    test_dupes = test_clean.duplicated().sum()
    print(f"\nDuplicate rows - Train: {train_dupes}, Test: {test_dupes}")
    
    if train_dupes > 0:
        train_clean.drop_duplicates(inplace=True)
        print(f"Removed {train_dupes} duplicates from train")
    
    return train_clean, test_clean

def handle_outliers(train_clean, test_clean):
    """Detect and handle outliers"""
    print("\n" + "="*50)
    print("HANDLING OUTLIERS")
    print("="*50)
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Handle Age outliers
    Q1_age = train_clean['Age'].quantile(0.25)
    Q3_age = train_clean['Age'].quantile(0.75)
    IQR_age = Q3_age - Q1_age
    lower_age = Q1_age - 1.5 * IQR_age
    upper_age = Q3_age + 1.5 * IQR_age
    
    age_outliers = train_clean[(train_clean['Age'] < lower_age) | (train_clean['Age'] > upper_age)]
    print(f"\nAge - Outliers detected: {len(age_outliers)} ({len(age_outliers)/len(train_clean)*100:.1f}%)")
    
    # Visualize Age before
    axes[0,0].hist(train_clean['Age'], bins=30, edgecolor='black')
    axes[0,0].axvline(lower_age, color='r', linestyle='--', label='Lower bound')
    axes[0,0].axvline(upper_age, color='r', linestyle='--', label='Upper bound')
    axes[0,0].set_title('Age Distribution (Before)')
    axes[0,0].set_xlabel('Age')
    axes[0,0].legend()
    
    # Cap Age outliers
    train_clean['Age'] = train_clean['Age'].clip(lower_age, upper_age)
    test_clean['Age'] = test_clean['Age'].clip(lower_age, upper_age)
    
    # Visualize Age after
    axes[0,1].hist(train_clean['Age'], bins=30, edgecolor='black')
    axes[0,1].set_title('Age Distribution (After Capping)')
    axes[0,1].set_xlabel('Age')
    
    # Handle Fare outliers
    Q1_fare = train_clean['Fare'].quantile(0.25)
    Q3_fare = train_clean['Fare'].quantile(0.75)
    IQR_fare = Q3_fare - Q1_fare
    lower_fare = Q1_fare - 1.5 * IQR_fare
    upper_fare = Q3_fare + 1.5 * IQR_fare
    
    fare_outliers = train_clean[train_clean['Fare'] > upper_fare]
    print(f"Fare - Outliers detected: {len(fare_outliers)} ({len(fare_outliers)/len(train_clean)*100:.1f}%)")
    
    # Visualize Fare before
    axes[1,0].hist(train_clean['Fare'], bins=30, edgecolor='black')
    axes[1,0].axvline(upper_fare, color='r', linestyle='--', label='Upper bound')
    axes[1,0].set_title('Fare Distribution (Before)')
    axes[1,0].set_xlabel('Fare')
    axes[1,0].legend()
    
    # Cap Fare outliers (only upper bound)
    train_clean['Fare'] = train_clean['Fare'].clip(0, upper_fare)
    test_clean['Fare'] = test_clean['Fare'].clip(0, upper_fare)
    
    # Visualize Fare after
    axes[1,1].hist(train_clean['Fare'], bins=30, edgecolor='black')
    axes[1,1].set_title('Fare Distribution (After Capping)')
    axes[1,1].set_xlabel('Fare')
    
    plt.tight_layout()
    plt.savefig('notebooks/outlier_treatment.png')
    
    return train_clean, test_clean

def main():
    """Main function to run the cleaning pipeline"""
    print("="*60)
    print("TITANIC DATA CLEANING PIPELINE")
    print("="*60)
    
    # Load data
    train, test = load_data('data/train.csv', 'data/test.csv')
    
    # Analyze missing values
    analyze_missing_values(train, "Training Data")
    analyze_missing_values(test, "Test Data")
    
    # Handle missing values
    train_clean, test_clean = handle_missing_values(train, test)
    
    # Check consistency
    train_clean, test_clean = check_consistency(train_clean, test_clean)
    
    # Handle outliers
    train_clean, test_clean = handle_outliers(train_clean, test_clean)
    
    # Save cleaned data
    train_clean.to_csv('data/train_cleaned.csv', index=False)
    test_clean.to_csv('data/test_cleaned.csv', index=False)
    print("\n" + "="*60)
    print("CLEANING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Cleaned train shape: {train_clean.shape}")
    print(f"Cleaned test shape: {test_clean.shape}")
    print("\nFiles saved:")
    print("  - data/train_cleaned.csv")
    print("  - data/test_cleaned.csv")

if __name__ == "__main__":
    main()