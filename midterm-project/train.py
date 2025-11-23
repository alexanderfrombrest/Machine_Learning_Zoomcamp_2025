import pandas as pd
import ast
import numpy as np
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import xgboost as xgb
import pickle

data = 'mazowieckie-spring25.csv'
df_final_cleaned = transform(data)

# 1. Split the raw dataframe
df_train, df_test = train_test_split(df_final_cleaned, test_size=0.2, random_state=1)
y_train = df_train['price_log'].reset_index(drop=True)
y_test = df_test['price_log'].reset_index(drop=True)

cols_to_drop = ['price', 'price_log']

X_train = df_train.drop(columns=cols_to_drop).reset_index(drop=True)
X_test = df_test.drop(columns=cols_to_drop).reset_index(drop=True)

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


encoder = ce.TargetEncoder(cols=['location_district'], handle_unknown='value', handle_missing='value')
encoder.fit(X_train, y_train)

X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)
current_feature_names = X_train_encoded.columns.tolist()

dtrain = xgb.DMatrix(X_train_encoded, label=y_train, feature_names=current_feature_names)
dtest = xgb.DMatrix(X_test_encoded, label=y_test, feature_names=current_feature_names)

xgb_params = {
    'eta': 0.05,              # Lower learning rate (makes learning slower but more robust)
    'max_depth': 5,           # Shallower trees
    'min_child_weight': 5,    # Conservative: needs 5 samples to make a split
    'subsample': 0.8,         # Randomness to prevent overfitting
    'colsample_bytree': 0.8,  # Randomness to prevent overfitting
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

watchlist = [(dtrain, 'train'), (dtest, 'test')]

# Since eta is lower (0.05), we need more rounds
model = xgb.train(
    xgb_params, 
    dtrain, 
    num_boost_round=1000,
    evals=watchlist,
    verbose_eval=50
)

# Save the model
output_file = 'model_pipeline.bin'

f_out = open(output_file, 'wb')
pickle.dump((encoder,model), f_out)
f_out.close()
