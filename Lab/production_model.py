import pandas as pd
import joblib

# Läs in testdatan
test_samples = pd.read_csv('../Lab/data/test_samples.csv')

# Läs in modellen från .pkl-filen
model = joblib.load('trained_model.pkl')
# Extrahera funktioner (features) från testdatan
X_test = test_samples.drop('cardio', axis=1)

# Gör predictions på testdatan
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Skapa en DataFrame med resultatet
result_df = pd.DataFrame({
    'probability class 0': probabilities[:, 0],
    'probability class 1': probabilities[:, 1],
    'prediction': predictions
})

# Exportera resultatet till en CSV-fil
result_df.to_csv('../Lab/data/prediction.csv', index=False)
