import pickle

import pandas as pd
import numpy as np
import uvicorn

from typing import Optional, Union, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transform import transform

# request
class Property(BaseModel):
    # Core identifying fields
    id: Optional[float] = None
    area: float
    roomsNum: Union[int, str]  # Raw data often sends "3" as a string
    
    # Location data
    location_latitude: float
    location_longitude: float
    location_district: Optional[str] = None
    location_city: Optional[str] = None
    
    # Building details
    buildYear: Optional[float] = None
    buildingFloorsNumber: Optional[float] = None
    floorNumber: Optional[str] = None
    constructionStatus: Optional[str] = None
    buildingOwnership: Optional[str] = None
    
    # Market and Type
    market: Optional[str] = "PRIMARY"
    userType: Optional[str] = "agency"
    propertyType: Optional[str] = "mieszkanie"
    
    # These are often lists-as-strings in raw data
    features: Optional[Union[str, List[str]]] = None
    extrasTypes: Optional[Union[str, List[str]]] = None
    securityTypes: Optional[Union[str, List[str]]] = None
    
    # Allow extra fields so the API doesn't crash if 
    # Otodom adds new fields to their JSON
    class Config:
        extra = "allow"


# response
class PredictResponce(BaseModel):
    predicted_price_pln: float

# API created in FastAPI and exposed on port 9696
app = FastAPI(title='price-prediction')

with open('model_pipeline.bin', 'rb') as f_in:
    model_pipeline = pickle.load(f_in)


@app.post("/predict")
def predict(property_json: Property) -> PredictResponce:
  data_dict = property_json.model_dump()

  property = pd.DataFrame([data_dict])
  property_cleaned = transform(property).drop(columns=['price', 'price_log'], errors='ignore').reset_index(drop=True)

  # Get the features the model is actually looking for
  model_features = model_pipeline.feature_names_in_

  # Ensure test_df_cleaned has ONLY those columns and in the SAME order
  X_test = property_cleaned.reindex(columns=model_features, fill_value=0)

  log_prediction = model_pipeline.predict(X_test)[0]
  prediction = np.expm1(log_prediction)

  print(f"Predicted Fair Value: {prediction:,.0f} PLN")
  return PredictResponce(
      predicted_price_pln = prediction
  )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9696)