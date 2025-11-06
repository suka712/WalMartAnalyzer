import joblib
import pandas as pd
import numpy as np
import os
import json

class WalmartSalesPredictor:
    def __init__(self, model_dir):
        """Loads all necessary artifacts when the predictor is created."""
        print(f"[Predictor] Initializing and loading artifacts from: {model_dir}")
        self.model_dir = model_dir
        
        with open(os.path.join(self.model_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        self.best_model_name = self.metadata.get('model_name', 'Unknown')
        print(f"[Predictor] Best model type: {self.best_model_name}")

        self.selected_features = joblib.load(os.path.join(self.model_dir, 'selected_features.pkl'))
        self.store_dept_baselines = joblib.load(os.path.join(self.model_dir, 'store_dept_baselines.pkl'))
        self.holiday_lifts = joblib.load(os.path.join(self.model_dir, 'holiday_lifts.pkl'))
        
        if self.best_model_name == 'Ridge':
             self.pipeline = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
             self.preprocessor = None
        elif self.best_model_name in ['LightGBM', 'XGBoost', 'LSTM']:
             self.model = joblib.load(os.path.join(self.model_dir, 'model.pkl'))
             self.preprocessor = joblib.load(os.path.join(self.model_dir, 'preprocessor.pkl'))
             self.pipeline = None
        else:
             raise ValueError(f"Unknown model type '{self.best_model_name}' in metadata.")
        
        print("[Predictor] Artifacts loaded successfully.")

    def _preprocess_input(self, input_data):
        """Applies feature engineering steps to new input data."""
        df = input_data.copy()
        
        # --- 1. Basic Feature Engineering ---
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
        df['Week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)

        for col in ['Is_SuperBowl', 'Is_Thanksgiving', 'Is_Christmas']:
            if col not in df.columns: df[col] = 0

        # --- FIX: Robustly handle potentially missing columns ---
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for col in markdown_cols:
            if col not in df:
                df[col] = 0 # If the column is completely missing, create it with 0
            else:
                df[col] = df[col].fillna(0) # If the column exists, just fill NaNs
        # --- END FIX ---
                 
        df['Promo_Active'] = (df[markdown_cols].gt(0).any(axis=1)).astype(int)
        df['Promo_Count'] = df[markdown_cols].gt(0).sum(axis=1)
        df['Total_Markdown'] = df[markdown_cols].sum(axis=1)

        # --- FIX: Apply the same robust logic to time-series columns ---
        time_series_cols = [f'Lag_{i}' for i in [1, 2, 4, 52]] + \
                           [f'Rolling_Avg_{w}' for w in [4, 8, 12]] + \
                           [f'Rolling_Std_{w}' for w in [4, 8, 12]] + \
                           ['Sales_Momentum', 'Store_Total_Sales']
        for col in time_series_cols:
            if col not in df:
                df[col] = 0 # If the column is completely missing, create it with 0
            else:
                df[col] = df[col].fillna(0) # If the column exists, just fill NaNs
        # --- END FIX ---
        
        # --- 3. Robustly Merge and Create Baseline Features ---
        print("[Predictor] Merging store-department baselines...")
        df['Store'] = df['Store'].astype(int)
        df['Dept'] = df['Dept'].astype(int)
        self.store_dept_baselines['Store'] = self.store_dept_baselines['Store'].astype(int)
        self.store_dept_baselines['Dept'] = self.store_dept_baselines['Dept'].astype(int)
        
        df = pd.merge(df, self.store_dept_baselines, on=['Store', 'Dept'], how='left')

        baseline_cols = ['StoreDept_Mean', 'StoreDept_Std']
        for col in baseline_cols:
            if df[col].isnull().any():
                fill_value = self.store_dept_baselines[col].mean() if col == 'StoreDept_Mean' else 0
                df[col] = df[col].fillna(fill_value)
        
        # --- 4. Create Dependent Features AFTER Merging ---
        df['Sales_vs_Baseline'] = df['Lag_1'] / (df['StoreDept_Mean'] + 1)
        df['Dept_Share_of_Store'] = df['Lag_1'] / (df['Store_Total_Sales'] + 1)
        
        # --- 5. Create Remaining Interaction Features ---
        df['Size_x_Unemployment'] = df['Size'] * df['Unemployment']
        
        self.holiday_lifts['Dept'] = self.holiday_lifts['Dept'].astype(int)
        df = pd.merge(df, self.holiday_lifts, on='Dept', how='left')
        df['Holiday_Lift'] = df['Holiday_Lift'].fillna(1.0)
        df['Dept_Holiday_Expected_Lift'] = df['Holiday_Lift'] * df['IsHoliday']

        # --- 6. Final Selection and Cleanup ---
        for f in self.selected_features:
            if f not in df.columns:
                df[f] = 0
        
        X = df[self.selected_features].copy()
        
        categorical_features_in_model = [f for f in self.selected_features if f in ['Store', 'Dept', 'Type', 'IsHoliday']]
        for col in categorical_features_in_model:
            if col in X.columns:
                 X[col] = X[col].astype('category')
                 
        return X

    def predict(self, input_data):
        """Predicts weekly sales for input data."""
        # ... (The predict method itself does not need to be changed) ...
        print(f"[Predictor] Received {len(input_data)} rows for prediction.")
        X = self._preprocess_input(input_data)
        
        if self.pipeline:
            pred_transformed = self.pipeline.predict(X)
        else:
            X_processed = self.preprocessor.transform(X)
            if self.best_model_name == 'LSTM':
                X_processed = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))
            pred_transformed = self.model.predict(X_processed)
        
        predictions = np.expm1(pred_transformed.flatten())
        predictions = np.maximum(predictions, 0)
        
        output = input_data[['Store', 'Dept', 'Date']].copy()
        output['Predicted_Weekly_Sales'] = predictions
        
        print(f"[Predictor] Prediction complete.")
        return output