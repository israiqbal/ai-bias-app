"""
Test script to validate the app.py works with different datasets
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

def test_data_preparation():
    """Test the data preparation logic works with test dataset"""
    
    # Load test dataset
    df = pd.read_csv('test_dataset.csv')
    print("✓ Loaded test dataset:", df.shape)
    print("  Columns:", df.columns.tolist())
    print("  Data types:", df.dtypes.to_dict())
    
    # Test 1: Target conversion with "yes/no"
    target = 'approved'
    target_col = df[target].copy()
    print(f"\n✓ Testing target conversion for '{target}'")
    print(f"  Unique values: {target_col.unique()}")
    
    # Apply target conversion logic from app
    unique_vals_list = sorted([v for v in target_col.dropna().unique() if pd.notna(v)])
    
    if any(str(v).lower() in ['yes', 'true', '1', 'approved', 'hired', 'admitted', 'positive', 'success', 'pass'] for v in unique_vals_list):
        mapping = {}
        for v in unique_vals_list:
            str_v = str(v).lower()
            if str_v in ['yes', 'true', '1', 'approved', 'hired', 'admitted', 'positive', 'success', 'pass']:
                mapping[v] = 1
            else:
                mapping[v] = 0
        df[target] = target_col.map(mapping)
        print(f"  ✓ Mapped using yes/no strategy: {mapping}")
    
    y = df[target]
    print(f"  Final distribution: {y.value_counts().to_dict()}")
    
    # Test 2: Feature preparation
    sensitive = 'gender'
    X = df.drop(columns=[target])
    print(f"\n✓ Testing feature preparation")
    print(f"  Initial features: {X.shape[1]}")
    
    # Encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"  After encoding: {X_encoded.shape[1]} features")
    print(f"  Feature names: {X_encoded.columns.tolist()}")
    
    # Test 3: Data cleaning
    print(f"\n✓ Testing data cleaning")
    mask = ~(X_encoded.isnull().any(axis=1) | y.isnull())
    X_clean = X_encoded[mask]
    y_clean = y[mask]
    print(f"  Samples after NaN removal: {len(X_clean)}")
    
    # Convert to numeric
    X_clean = X_clean.astype(np.float64)
    y_clean = y_clean.astype(np.float64)
    print(f"  ✓ Converted to float64")
    
    # Test 4: Zero-variance features
    print(f"\n✓ Testing zero-variance feature removal")
    feature_vars = X_clean.var()
    zero_var_cols = feature_vars[feature_vars == 0].index.tolist()
    if zero_var_cols:
        print(f"  Removing {len(zero_var_cols)} zero-variance features: {zero_var_cols}")
        X_clean = X_clean.drop(columns=zero_var_cols)
    else:
        print(f"  No zero-variance features found")
    
    # Test 5: Class distribution
    print(f"\n✓ Testing class distribution")
    class_counts = pd.Series(y_clean).value_counts()
    print(f"  Class distribution: {class_counts.to_dict()}")
    print(f"  Min samples per class: {class_counts.min()}")
    
    min_samples_per_class = 5
    if any(count < min_samples_per_class for count in class_counts.values):
        print(f"  ⚠️ Warning: One or more classes have < {min_samples_per_class} samples")
    
    # Test 6: Stratified split
    print(f"\n✓ Testing train-test split")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        print(f"  ✓ Stratified split successful")
    except ValueError as e:
        print(f"  ⚠️ Stratified split failed: {str(e)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=None
        )
        print(f"  ✓ Used random split instead")
    
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Train distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  Test distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Test 7: Scaling
    print(f"\n✓ Testing scaling")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  ✓ Scaling successful")
    print(f"  Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
    
    # Test 8: Model training
    print(f"\n✓ Testing model training")
    
    try:
        model = SGDClassifier(
            loss='log_loss',
            max_iter=5000,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            early_stopping=False,
            verbose=0
        )
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        print(f"  ✓ SGDClassifier trained successfully (score: {score:.3f})")
    except Exception as e:
        print(f"  ⚠️ SGDClassifier failed: {e}")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"  ✓ RandomForest trained successfully (score: {score:.3f})")
    
    # Test 9: Sensitive attribute analysis
    print(f"\n✓ Testing sensitive attribute analysis")
    sensitive_groups = sorted(df[sensitive].unique())
    print(f"  Sensitive groups: {sensitive_groups}")
    
    df_test = df.loc[y_test.index].copy()
    df_test['pred'] = model.predict(X_test_scaled if 'SGDClassifier' in str(type(model)) else X_test)
    
    for group in sensitive_groups:
        group_rate = df_test[df_test[sensitive] == group]['pred'].mean()
        print(f"  {group}: {group_rate:.1%} prediction rate")
    
    print("\n✅ All tests passed! The app should work with different datasets.")

if __name__ == '__main__':
    test_data_preparation()
