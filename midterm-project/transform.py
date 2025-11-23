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
import numpy as np


pd.set_option('display.max_columns', None)


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in kilometers between two points
    (specified in decimal degrees) on the earth.

    This is a vectorized implementation using NumPy.

    Parameters:
    lat1, lon1: NumPy arrays of latitudes and longitudes for the first set of points.
    lat2, lon2: NumPy arrays (or single values) for the second set of points.

    Returns:
    A NumPy array containing the distances in kilometers.
    """
    # Mean Earth radius in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance_km = R * c
    return distance_km


def safe_convert_to_list(x):
    if pd.isna(x):
        return np.nan # Keep NaN as NaN
    try:
        # This safely evaluates the string as a Python literal (a list)
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        # If it's not a valid list string (e.g., 'not a list')
        return np.nan # Treat bad data as missing


df = pd.read_csv('mazowieckie-spring25.csv', encoding ='utf-8')

# Filter only Warsaw
df = df[df['city'] == 'warszawa']
df.reset_index(inplace=True)


features_and_target = [
    # --- Group 0: Target variable
    'price',

    # --- Group 1: Numerical Features
    'area',
    'buildYear',
    'buildingFloorsNumber',
    'floorNumber',
    'roomsNum',
    'location_latitude',
    'location_longitude',

    # --- Group 2: Categorical Features (Need Encoding)
    # We need to encode them (e.g., One-Hot, Target, or Label Encoding).
    'market',
    'buildingMaterial',
    'constructionStatus',
    'ownership',
    'userType',
    'location_district',

    # --- Group 3: List/Text Features (Need Engineering)
    'features'
]

df = df[features_and_target]

# Warsaw city center coordinates to calculate distance from center
WARSAW_CENTER_LAT = 52.2286
WARSAW_CENTER_LON = 21.0031

df['distance_from_center'] = haversine_vectorized(df['location_latitude'], df['location_longitude'], WARSAW_CENTER_LAT, WARSAW_CENTER_LON)
df['features'] = df['features'].apply(safe_convert_to_list)


# Features columns has many features of property. We should take the most important of them (features_to_engineer).
# After that we should use multi-label binarization technique, since one property has many features. We will encode all the important features as 0/1 columns using **MultiLabelBinarizer**

features_to_engineer = [
    'taras',
    'ogródek',
    'winda',
    'balkon',
    'klimatyzacja',
    'pom. użytkowe',
    'piwnica',
    'dwupoziomowe',
    'garaż/miejsce parkingowe',
    'oddzielna kuchnia',
    'teren zamknięty'
]

mlb = MultiLabelBinarizer()

encoded_df = pd.DataFrame(mlb.fit_transform(df['features']), columns=mlb.classes_, index=df.index)

encoded_df = encoded_df[features_to_engineer]
df_final = pd.concat([df, encoded_df], axis=1)
df_final = df_final.drop(columns=['features'])


# Now lets fix the values of floorNumber column
df_final.floorNumber.unique()
df_final.buildingFloorsNumber.unique()

# Map the floor strings to numeric values


floor_map = {
    'cellar': -1,
    'ground_floor': 0,
    'floor_1': 1,
    'floor_2': 2,
    'floor_3': 3,
    'floor_4': 4,
    'floor_5': 5,
    'floor_6': 6,
    'floor_7': 7,
    'floor_8': 8,
    'floor_9': 9,
    'floor_10': 10,
    'floor_higher_10': 11  # Using 11 to maintain the order (it's > 10)
}

df_final['floor_numeric'] = df_final['floorNumber'].map(floor_map)
df_final = df_final.drop(columns=['floorNumber'])
df_final['floor_numeric'] = df_final['floor_numeric'].astype('float64')
median_floor = df_final['floor_numeric'].median(skipna=True)
df_final['floor_numeric'] = df_final['floor_numeric'].fillna(median_floor)



df_final = df_final.dropna(subset=['price']).reset_index(drop=True)


df_final['roomsNum'] = df_final['roomsNum'].replace('more', 11)
df_final['roomsNum'] = pd.to_numeric(df_final['roomsNum'], errors='coerce')

# 1. Define the logical order
# We assign numbers based on value. 'nan' can be a neutral 0.
status_map = {
    'to_renovation': -1,  # Lowest value
    'to_completion': 0,   # Neutral/Primary market
    'ready_to_use': 1,    # Highest value
    np.nan: 0             # Assign 'nan' to the neutral category
}

# 2. Map the values and fill any that were missed (just in case)
df_final['constructionStatus_numeric'] = df_final['constructionStatus'].map(status_map)
df_final['constructionStatus_numeric'].fillna(0, inplace=True)

# 3. Drop the original column
df_final = df_final.drop('constructionStatus', axis=1)


# 3. The High-Cardinality Column: location_district
# Your location_district column is also an object and likely has many unique values (e.g., 20+ districts).
# 
# Bad Method: One-Hot Encoding (pd.get_dummies) would create 20+ new columns, which is inefficient.
# 
# Best Method: Target Encoding.
# 
# Target Encoding is a powerful technique that replaces the district's name with a number. The number it uses is the average price for that district. This is an extremely strong signal for your model.
# 
# The easiest way to do this is with the category_encoders library.


# 1. List the columns you want to encode
columns_to_onehot = ['market', 'buildingMaterial', 'userType', 'ownership']

# 2. Use pd.get_dummies
# dummy_na=True is a great trick! It will create a new column 
# like 'buildingMaterial_nan', which is a very strong signal.
df_final = pd.get_dummies(
    df_final, 
    columns=columns_to_onehot, 
    dummy_na=True,  # Creates a column for 'nan' values
    drop_first=False  # Keeps all categories
)


median_year = df_final['buildYear'].median(skipna=True)
median_building_floors_number = df_final['buildingFloorsNumber'].median(skipna=True)

df_final['buildYear'] = df_final['buildYear'].fillna(median_year)
df_final['buildingFloorsNumber'] = df_final['buildingFloorsNumber'].fillna(median_building_floors_number)



df_final_cleaned = df_final.copy()


low = df_final_cleaned['price'].quantile(0.01)
high = df_final_cleaned['price'].quantile(0.965)
print((low, high))

# Values are quite reasonable: on market we rarely see flats below 400k, for Warsaw is even 500k.
# As well as 5M flats are quite rare. We can even take 3M, lets remove 3%+ the most expensive.

df_final_cleaned = df_final_cleaned[(df_final_cleaned['price']>low) & (df_final_cleaned['price']<high)]

# Convert price to log(price)
df_final_cleaned.reset_index(drop=True)
df_final_cleaned['price_log'] = np.log1p(df_final_cleaned['price'])