import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# --- Configuration & Constants ---

WARSAW_CENTER_LAT = 52.2286
WARSAW_CENTER_LON = 21.0031

# Features that should exist in the dataframe (excluding target for safety)
# Note: 'price' is handled separately to allow this function to work for inference.
BASE_FEATURES = [
    'area', 'buildYear', 'buildingFloorsNumber', 'floorNumber', 'roomsNum',
    'location_latitude', 'location_longitude', 'market', 'buildingMaterial',
    'constructionStatus', 'ownership', 'userType', 'location_district', 'features'
]

FEATURES_TO_ENGINEER = [
    'taras', 'ogródek', 'winda', 'balkon', 'klimatyzacja', 'pom. użytkowe',
    'piwnica', 'dwupoziomowe', 'garaż/miejsce parkingowe',
    'oddzielna kuchnia', 'teren zamknięty'
]

FLOOR_MAP = {
    'cellar': -1, 'ground_floor': 0, 'floor_1': 1, 'floor_2': 2, 'floor_3': 3,
    'floor_4': 4, 'floor_5': 5, 'floor_6': 6, 'floor_7': 7, 'floor_8': 8,
    'floor_9': 9, 'floor_10': 10, 'floor_higher_10': 11
}

STATUS_MAP = {
    'to_renovation': -1,
    'to_completion': 0,
    'ready_to_use': 1,
    np.nan: 0
}

COLUMNS_TO_ONEHOT = ['market', 'buildingMaterial', 'userType', 'ownership']


# --- Helper Functions ---

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in kilometers between two points using NumPy.
    """
    R = 6371.0 # Radius of Earth in km
    
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def safe_convert_to_list(x):
    """
    Safely evaluates a string as a Python list. Returns NaN on failure.
    """
    if pd.isna(x):
        return np.nan
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return np.nan


# --- Main Transformation Logic ---

def transform(data: pd.DataFrame, drop_outliers: bool = False):
    """
    Cleans, engineers features, and prepares the dataframe for training or inference.
    
    Args:
        data (pd.DataFrame): Raw input data.
        drop_outliers (bool): If True, drops price outliers. Should be False during inference.
    
    Returns:
        pd.DataFrame: Processed dataframe ready for the model.
    """


    EXPECTED_COLUMNS = ['area', 'buildYear', 'buildingFloorsNumber', 'roomsNum',
       'location_latitude', 'location_longitude', 'location_district',
       'distance_from_center', 'taras', 'ogródek', 'winda', 'balkon',
       'klimatyzacja', 'pom. użytkowe', 'piwnica', 'dwupoziomowe',
       'garaż/miejsce parkingowe', 'oddzielna kuchnia', 'teren zamknięty',
       'floor_numeric', 'constructionStatus_numeric', 'market_PRIMARY',
       'market_SECONDARY', 'market_nan', 'buildingMaterial_breezeblock',
       'buildingMaterial_brick', 'buildingMaterial_cellular_concrete',
       'buildingMaterial_concrete', 'buildingMaterial_concrete_plate',
       'buildingMaterial_hydroton', 'buildingMaterial_other',
       'buildingMaterial_reinforced_concrete', 'buildingMaterial_silikat',
       'buildingMaterial_wood', 'buildingMaterial_nan', 'userType_agency',
       'userType_developer', 'userType_private', 'userType_nan',
       'ownership_full_ownership', 'ownership_limited_ownership',
       'ownership_share', 'ownership_usufruct', 'ownership_nan']


    # 1. Avoid modifying the original dataframe
    df = data.copy()

    # 2. Filter Scope (Warsaw only)
    # We use .get() just in case 'city' is missing in future inputs, though expected
    if 'city' in df.columns:
        df = df[df['city'] == 'warszawa'].copy()
    
    # 3. Select relevant columns
    # We include 'price' only if it exists in the input
    cols_to_keep = BASE_FEATURES.copy()
    if 'price' in df.columns:
        cols_to_keep.append('price')
    
    # Filter columns that actually exist in the input to prevent KeyErrors
    cols_present = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_present]

    # 4. Feature Engineering: Location Distance
    df['distance_from_center'] = haversine_vectorized(
        df['location_latitude'], df['location_longitude'], 
        WARSAW_CENTER_LAT, WARSAW_CENTER_LON
    )

    # 5. Feature Engineering: "Features" column (MultiLabelBinarizer)
    df['features'] = df['features'].apply(safe_convert_to_list)
    
    # Initialize MLB
    mlb = MultiLabelBinarizer()
    # We fit transform, but we must ensure we only keep the ones we care about
    # Logic: fit on current batch, but then reindex to ensure consistent columns
    features_encoded = pd.DataFrame(
        mlb.fit_transform(df['features'].dropna()),
        columns=mlb.classes_,
        index=df.index[df['features'].notna()]
    )
    
    # If a row had NaN features, fill with 0
    features_encoded = features_encoded.reindex(df.index, fill_value=0)

    # Keep only the specific engineered features we want; fill missing columns with 0
    for col in FEATURES_TO_ENGINEER:
        if col not in features_encoded.columns:
            features_encoded[col] = 0
    
    df = pd.concat([df, features_encoded[FEATURES_TO_ENGINEER]], axis=1)
    df = df.drop(columns=['features'])

    # 6. Numeric Mapping & Cleaning
    
    # Floor Number
    df['floor_numeric'] = df['floorNumber'].map(FLOOR_MAP)
    df['floor_numeric'] = pd.to_numeric(df['floor_numeric'], errors='coerce')
    # Simple imputation (ideally should be calculated on train set)
    df['floor_numeric'] = df['floor_numeric'].fillna(df['floor_numeric'].median())
    df = df.drop(columns=['floorNumber'])

    # Rooms
    df['roomsNum'] = df['roomsNum'].replace('more', 11)
    df['roomsNum'] = pd.to_numeric(df['roomsNum'], errors='coerce')

    # Construction Status
    df['constructionStatus_numeric'] = df['constructionStatus'].map(STATUS_MAP)
    df['constructionStatus_numeric'] = df['constructionStatus_numeric'].fillna(0)
    df = df.drop(columns=['constructionStatus'])

    # Impute other numerics
    df['buildYear'] = df['buildYear'].fillna(df['buildYear'].median())
    df['buildingFloorsNumber'] = df['buildingFloorsNumber'].fillna(df['buildingFloorsNumber'].median())

    # 7. One-Hot Encoding (Categorical)
    df = pd.get_dummies(
        df, 
        columns=COLUMNS_TO_ONEHOT, 
        dummy_na=True, 
        drop_first=False
    )

    if EXPECTED_COLUMNS is not None:
        # Add missing columns with zeros
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = 0 

    # 8. Target Handling (Price)
    if 'price' in df.columns:
        # Drop rows with no price info
        df = df.dropna(subset=['price'])
        
        # Outlier Removal (Only if requested, typically for Training)
        if drop_outliers:
            low = df['price'].quantile(0.01)
            high = df['price'].quantile(0.965)
            df = df[(df['price'] > low) & (df['price'] < high)]

        # Log Transform
        df['price_log'] = np.log1p(df['price'])
        
        # Reset index after filtering
        df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Test the function if run directly
    print("Running transform on local CSV...")
    try:
        raw_df = pd.read_csv('mazowieckie-spring25.csv', encoding='utf-8')
        processed_df = transform(raw_df, drop_outliers=True)
        print("Transformation successful!")
        print(f"Original shape: {raw_df.shape}")
        print(f"Processed shape: {processed_df.shape}")
        print(processed_df.head(3))
    except FileNotFoundError:
        print("CSV file not found. Please ensure 'mazowieckie-spring25.csv' is in the folder.")