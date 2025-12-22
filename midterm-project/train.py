import pickle

import pandas as pd
import numpy as np
import category_encoders as ce
import xgboost as xgb

from sklearn.pipeline import Pipeline
from transform import transform


def load_data(filename):
    # 1. Load the training dataset from .csv file
    data = pd.read_csv(filename)
    df_final_cleaned = transform(data)
    return df_final_cleaned

def train_model(df_final_cleaned):

    y_train = df_final_cleaned['price_log'].reset_index(drop=True)
    X_train = df_final_cleaned.drop(columns=['price', 'price_log']).reset_index(drop=True)

    # ### Method Note: Handling High-Cardinality Features (District)
    # 
    # **Why TargetEncoder instead of One-Hot (DictVectorizer)?**
    # 
    # 1.  **The Cardinality Problem:** `location_district` has many unique values (high cardinality). One-Hot Encoding would create a new column for every district, resulting in a wide, sparse matrix that dilutes the signal.
    # 2.  **How TargetEncoder Works:** It replaces the category name (e.g., "Mokotów") with the average target value (mean price) for that category. It keeps the feature in **one single column**.
    # 3.  **XGBoost Efficiency:** Instead of needing 20+ branches to ask "Is it Mokotów?", "Is it Wola?", the tree can make one powerful split based on value (e.g., `district_value > 600k`), instantly separating expensive areas from cheaper ones.
    # 4.  **CRITICAL:** To prevent **Data Leakage**, we must split the data into Train/Test *before* fitting the TargetEncoder. The encoder must only learn average prices from the Training set.

    # We use handle_unknown='value' and handle_missing='value'
    # This handles cases where a district in the val/test set 
    # was not seen in the train set.

    xgb_params = {
        'learning_rate': 0.05,    # Lower learning rate (makes learning slower but more robust)
        'max_depth': 5,           # Shallower trees
        'min_child_weight': 5,    # Conservative: needs 5 samples to make a split
        'subsample': 0.8,         # Randomness to prevent overfitting
        'colsample_bytree': 0.8,  # Randomness to prevent overfitting
        'n_estimators': 1000,     # num_boost_round
        'objective': 'reg:squarederror',
        'n_jobs': 8,              # nthread
        'random_state': 1         # seed
    }

    model_pipeline = Pipeline(
        steps=[
            ('encoder', ce.TargetEncoder(cols=['location_district'], handle_unknown='value', handle_missing='value')),
            ('regressor', xgb.XGBRegressor(**xgb_params))   
        ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline


def save_model(model_pipeline):
    output_file = 'model_pipeline.bin'

    f_out = open(output_file, 'wb')
    pickle.dump((model_pipeline), f_out)
    f_out.close()

if __name__ == '__main__':
    df = load_data('mazowieckie-spring25.csv')
    model_pipeline = train_model(df)
    save_model(model_pipeline)
