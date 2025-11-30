import pickle
with open('model_pipeline.bin', 'rb') as f_in:
    (encoder, model) = pickle.load(f_in)

X = encoder.transform(test_row)

# Convert the Pandas DataFrame into a DMatrix
X_DMatrix = xgb.DMatrix(X, feature_names=model.feature_names)

log_prediction = model.predict(X_DMatrix)[0]

predicted_price_pln = np.expm1(log_prediction)
print(f"Predicted Fair Value: {predicted_price_pln:,.0f} PLN")