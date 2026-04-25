import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_family_features(df):
    """Create family-related features"""
    df_feat = df.copy()
    
    # Family Size
    df_feat['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone
    df_feat['IsAlone'] = (df_feat['FamilySize'] == 1).astype(int)
    
    return df_feat

def extract_title(df):
    """Extract title from Name"""
    df_feat = df.copy()
    
    # Extract title
    df_feat['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df_feat['Title'] = df_feat['Title'].map(title_mapping).fillna('Rare')
    
    return df_feat

def extract_deck(df):
    """Extract deck from Cabin (though we dropped it, this is for demonstration)"""
    df_feat = df.copy()
    
    # For demonstration, if Cabin existed, we'd extract first letter
    # Since we dropped Cabin, we'll create a placeholder
    df_feat['Deck'] = 'Unknown'
    
    return df_feat

def create_age_groups(df):
    """Create age groups"""
    df_feat = df.copy()
    
    # Define age groups
    bins = [0, 12, 18, 35, 60, 100]
    labels = ['Child', 'Teen', 'Adult', 'Middle_Aged', 'Senior']
    df_feat['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    return df_feat

def create_fare_features(df):
    """Create fare-related features"""
    df_feat = df.copy()
    
    # Fare per person
    df_feat['FarePerPerson'] = df['Fare'] / df_feat['FamilySize']
    
    # Log transform Fare (handle zeros)
    df_feat['Fare_Log'] = np.log1p(df['Fare'])
    
    return df_feat

def create_interaction_features(df):
    """Create interaction features"""
    df_feat = df.copy()
    
    # Pclass × Fare
    df_feat['Pclass_Fare'] = df['Pclass'] * df['Fare']
    
    # Age × Title (for important titles)
    title_importance = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df_feat['Title_Num'] = df_feat['Title'].map(title_importance)
    df_feat['Age_Title'] = df['Age'] * df_feat['Title_Num']
    
    return df_feat

def encode_features(df):
    """Encode categorical features"""
    df_encoded = df.copy()
    
    # One-hot encoding for nominal features
    nominal_features = ['Sex', 'Embarked', 'Title', 'AgeGroup']
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_features, prefix=nominal_features)
    
    # Ordinal encoding for Pclass
    df_encoded['Pclass_Ordinal'] = df_encoded['Pclass']
    
    return df_encoded

def scale_features(df, feature_cols):
    """Scale numerical features"""
    df_scaled = df.copy()
    
    scaler = StandardScaler()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df_scaled, scaler

def engineer_features(train, test):
    """Main function for feature engineering"""
    print("="*50)
    print("Starting Feature Engineering Process")
    print("="*50)
    
    # Apply all feature engineering steps
    train_feat = create_family_features(train)
    test_feat = create_family_features(test)
    
    train_feat = extract_title(train_feat)
    test_feat = extract_title(test_feat)
    
    train_feat = extract_deck(train_feat)
    test_feat = extract_deck(test_feat)
    
    train_feat = create_age_groups(train_feat)
    test_feat = create_age_groups(test_feat)
    
    train_feat = create_fare_features(train_feat)
    test_feat = create_fare_features(test_feat)
    
    train_feat = create_interaction_features(train_feat)
    test_feat = create_interaction_features(test_feat)
    
    # Visualize new features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Family Size distribution
    train_feat['FamilySize'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Family Size Distribution')
    
    # Title distribution
    train_feat['Title'].value_counts().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Title Distribution')
    
    # Age Group distribution
    train_feat['AgeGroup'].value_counts().plot(kind='bar', ax=axes[0,2])
    axes[0,2].set_title('Age Group Distribution')
    
    # Fare per person vs Survival
    if 'Survived' in train_feat.columns:
        sns.boxplot(x='Survived', y='FarePerPerson', data=train_feat, ax=axes[1,0])
        axes[1,0].set_title('Fare per Person vs Survival')
    
    # IsAlone vs Survival
    if 'Survived' in train_feat.columns:
        train_feat.groupby('IsAlone')['Survived'].mean().plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Survival Rate by IsAlone')
    
    # Title vs Survival
    if 'Survived' in train_feat.columns:
        train_feat.groupby('Title')['Survived'].mean().plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Survival Rate by Title')
    
    plt.tight_layout()
    plt.savefig('../notebooks/feature_engineering.png')
    plt.show()
    
    # Encode features
    train_encoded = encode_features(train_feat)
    test_encoded = encode_features(test_feat)
    
    # Drop original columns that are no longer needed
    cols_to_drop = ['Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch']
    if 'Survived' in train_encoded.columns:
        cols_to_drop.append('Survived')
    
    train_encoded = train_encoded.drop(columns=[c for c in cols_to_drop if c in train_encoded.columns])
    test_encoded = test_encoded.drop(columns=[c for c in cols_to_drop if c in test_encoded.columns])
    
    # Scale numerical features
    numerical_cols = train_encoded.select_dtypes(include=[np.number]).columns.tolist()
    if 'Survived' in numerical_cols:
        numerical_cols.remove('Survived')
    
    train_scaled, scaler = scale_features(train_encoded, numerical_cols)
    test_scaled, _ = scale_features(test_encoded, numerical_cols)
    
    # Add back Survived if it existed
    if 'Survived' in train_feat.columns:
        train_scaled['Survived'] = train_feat['Survived']
    
    print("\n" + "="*50)
    print(f"Feature Engineering Completed. New shape: {train_scaled.shape}")
    print("="*50)
    
    return train_scaled, test_scaled

if __name__ == "__main__":
    # Load cleaned data
    train_clean = pd.read_csv('../data/train_cleaned.csv')
    test_clean = pd.read_csv('../data/test_cleaned.csv')
    
    # Engineer features
    train_feat, test_feat = engineer_features(train_clean, test_clean)
    
    # Save engineered features
    train_feat.to_csv('../data/train_engineered.csv', index=False)
    test_feat.to_csv('../data/test_engineered.csv', index=False)
    print("\nEngineered data saved to data/train_engineered.csv and data/test_engineered.csv")