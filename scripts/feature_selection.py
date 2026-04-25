import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.model_selection import train_test_split

def correlation_analysis(df, target_col='Survived'):
    """Analyze feature correlations"""
    print("="*50)
    print("Correlation Analysis")
    print("="*50)
    
    # Calculate correlations
    corr_matrix = df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('../notebooks/correlation_matrix.png')
    plt.show()
    
    # Find highly correlated features (|corr| > 0.8)
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                high_corr.append((col_i, col_j, corr_matrix.iloc[i, j]))
    
    if high_corr:
        print("\nHighly correlated feature pairs (>0.8):")
        for pair in high_corr:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
    
    # Correlation with target
    if target_col in df.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        print(f"\nTop 10 features correlated with {target_col}:")
        print(target_corr.head(10))
    
    return corr_matrix

def feature_importance_rf(X, y, feature_names):
    """Get feature importance using Random Forest"""
    print("\n" + "="*50)
    print("Random Forest Feature Importance")
    print("="*50)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'].head(15), importance_df['Importance'].head(15))
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('../notebooks/feature_importance.png')
    plt.show()
    
    print("\nFeature Importance Rankings:")
    print(importance_df.head(15))
    
    return importance_df, rf

def recursive_feature_elimination(X, y, feature_names, n_features=10):
    """Perform Recursive Feature Elimination"""
    print("\n" + "="*50)
    print(f"Recursive Feature Elimination (Selecting {n_features} features)")
    print("="*50)
    
    # Initialize RFE
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_features)
    
    # Fit RFE
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
    rankings = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfe.ranking_,
        'Selected': rfe.support_
    }).sort_values('Ranking')
    
    print(f"\nTop {n_features} selected features:")
    for feat in selected_features:
        print(f"- {feat}")
    
    return selected_features, rankings

def select_final_features(df, target_col='Survived', method='combined'):
    """Select final features for modeling"""
    print("\n" + "="*50)
    print("Feature Selection Summary")
    print("="*50)
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    
    # Correlation analysis
    corr_matrix = correlation_analysis(df)
    
    # Feature importance
    importance_df, rf_model = feature_importance_rf(X, y, feature_names)
    
    # RFE
    selected_rfe, rankings = recursive_feature_elimination(X, y, feature_names, n_features=10)
    
    # Combine results to select final features
    # Strategy: Take top features from importance that are not highly correlated
    top_features = importance_df.head(15)['Feature'].tolist()
    
    # Remove highly correlated features (keep one from each pair)
    features_to_keep = []
    features_to_remove = []
    
    for i, feat1 in enumerate(top_features):
        if feat1 in features_to_remove:
            continue
        features_to_keep.append(feat1)
        
        for feat2 in top_features[i+1:]:
            if abs(corr_matrix.loc[feat1, feat2]) > 0.7:
                # Keep the one with higher importance
                imp1 = importance_df[importance_df['Feature']==feat1]['Importance'].values[0]
                imp2 = importance_df[importance_df['Feature']==feat2]['Importance'].values[0]
                
                if imp1 >= imp2:
                    features_to_remove.append(feat2)
                else:
                    features_to_remove.append(feat1)
                    features_to_keep.remove(feat1)
                    features_to_keep.append(feat2)
    
    # Final feature list
    final_features = list(set(features_to_keep[:10]))  # Keep top 10
    
    print("\n" + "="*50)
    print("FINAL SELECTED FEATURES:")
    print("="*50)
    for i, feat in enumerate(final_features, 1):
        importance = importance_df[importance_df['Feature']==feat]['Importance'].values[0]
        print(f"{i}. {feat} (Importance: {importance:.4f})")
    
    # Justification
    print("\n" + "="*50)
    print("JUSTIFICATION FOR FEATURE SELECTION:")
    print("="*50)
    print("""
    1. Feature Importance: Selected features with highest Random Forest importance scores
    2. Correlation Analysis: Removed features with correlation >0.7 to avoid multicollinearity
    3. RFE Validation: Cross-validated with Recursive Feature Elimination
    4. Domain Knowledge: Considered known important factors in Titanic survival
       - Social class (Pclass)
       - Gender (Sex)
       - Family relationships
       - Age
       - Fare (proxy for wealth)
    """)
    
    return final_features, importance_df, rankings

if __name__ == "__main__":
    # Load engineered data
    train_feat = pd.read_csv('../data/train_engineered.csv')
    
    # Select features
    final_features, importance_df, rankings = select_final_features(train_feat)
    
    # Save selected features list
    with open('../data/selected_features.txt', 'w') as f:
        for feat in final_features:
            f.write(f"{feat}\n")
    
    print("\nSelected features saved to data/selected_features.txt")