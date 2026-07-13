import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def auto_detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Detect whether task is 'classification' or 'regression' based on target type and unique counts."""
    target = df[target_col].dropna()
    unique_count = target.nunique()
    
    if pd.api.types.is_numeric_dtype(target):
        if unique_count <= 10:
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'

def preprocess_for_ml(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Impute, scale, and encode features and target column.
    Returns:
    - X_processed (numpy array)
    - y_processed (numpy array)
    - processed_feature_names (list)
    - metadata (dict with encodings information)
    """
    df_ml = df[[target_col] + feature_cols].dropna(subset=[target_col]).copy()
    
    y = df_ml[target_col].values
    X_raw = df_ml[feature_cols]
    
    # Handle Target Encoding if Classification
    target_encoder = None
    y_encoded = y
    is_classification = not pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique() <= 10
    
    if is_classification and not pd.api.types.is_integer_dtype(df[target_col]):
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))
    
    # Separate numeric and categorical features
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Impute numeric features
    X_num = X_raw[numeric_cols].values
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_num = num_imputer.fit_transform(X_num)
        
        # Scale numeric features
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
        
    # Encode and impute categorical features
    X_cat_list = []
    cat_feature_names = []
    
    for col in categorical_cols:
        col_data = X_raw[col].fillna("MISSING").astype(str).values
        # Use One-Hot Encoding via pandas get_dummies for categorical columns
        dummies = pd.get_dummies(col_data, prefix=col, drop_first=True)
        X_cat_list.append(dummies.values)
        cat_feature_names.extend(dummies.columns.tolist())
        
    # Combine processed features
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        X_processed = np.hstack([X_num, np.hstack(X_cat_list)])
        all_features = numeric_cols + cat_feature_names
    elif len(numeric_cols) > 0:
        X_processed = X_num
        all_features = numeric_cols
    elif len(categorical_cols) > 0:
        X_processed = np.hstack(X_cat_list)
        all_features = cat_feature_names
    else:
        raise ValueError("No features available to train models.")
        
    return X_processed, y_encoded, all_features, {
        "is_classification": is_classification,
        "target_classes": target_encoder.classes_.tolist() if target_encoder else None
    }

def train_and_evaluate(X: np.ndarray, y: np.ndarray, task_type: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Train several models and evaluate them.
    Returns model results containing metrics, estimators, and feature importance.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    results = {}
    
    if task_type == 'classification':
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state, max_depth=8),
            "Random Forest": RandomForestClassifier(random_state=random_state, n_estimators=100, max_depth=8),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state, n_estimators=100)
        }
        
        # Determine number of classes
        unique_classes = np.unique(y)
        is_binary = len(unique_classes) == 2
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Metrics
                acc = accuracy_score(y_test, preds)
                
                if is_binary:
                    prec = precision_score(y_test, preds, zero_division=0)
                    rec = recall_score(y_test, preds, zero_division=0)
                    f1 = f1_score(y_test, preds, zero_division=0)
                    
                    # ROC Curve
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        roc_auc = auc(fpr, tpr)
                        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
                    else:
                        roc_data = None
                else:
                    # Multi-class metrics
                    prec = precision_score(y_test, preds, average='macro', zero_division=0)
                    rec = recall_score(y_test, preds, average='macro', zero_division=0)
                    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
                    roc_data = None
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, preds).tolist()
                
                # Feature Importances
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_.tolist()
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_[0]).tolist()
                else:
                    importances = []
                    
                results[name] = {
                    "model": model,
                    "metrics": {
                        "Accuracy": float(acc),
                        "Precision": float(prec),
                        "Recall": float(rec),
                        "F1-Score": float(f1)
                    },
                    "confusion_matrix": cm,
                    "roc_curve": roc_data,
                    "feature_importances": importances
                }
            except Exception as e:
                # Log model failure gracefully
                results[name] = {"error": str(e)}
                
    else: # regression
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=8),
            "Random Forest": RandomForestRegressor(random_state=random_state, n_estimators=100, max_depth=8),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state, n_estimators=100)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                # Feature Importances
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_.tolist()
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_).tolist()
                else:
                    importances = []
                    
                results[name] = {
                    "model": model,
                    "metrics": {
                        "R2-Score (R-squared)": float(r2),
                        "Mean Squared Error (MSE)": float(mse),
                        "Mean Absolute Error (MAE)": float(mae)
                    },
                    "feature_importances": importances
                }
            except Exception as e:
                results[name] = {"error": str(e)}
                
    return results
