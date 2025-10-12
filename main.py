import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. App Creation & Model Loading ---
# Create the FastAPI application
app = FastAPI(
    title="Haldiram's Product Performance Predictor API",
    description="An API to predict if a new product is likely to be an underperformer.",
    version="1.0"
)

# Load the trained model pipeline from the saved file
try:
    with open("best_performing_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'best_performing_model.pkl' not found. Please place it in the same directory.")
    model = None

# --- 2. Pydantic Model for Input Validation ---
# This defines the exact structure and data types for the input JSON.
class ProductFeatures(BaseModel):
    rating: float
    price_whole: int
    mrp: int
    number_of_global_ratings: int
    number_of_reviews: int
    discount_percentage_cleaned: int
    product_weight_grams: int
    category_reclassified: str

# --- 3. API Endpoint Definition ---
@app.post("/predict", tags=["Prediction"])
def predict_performance(product_features: ProductFeatures):
    """
    Predicts if a product will be an underperformer based on its features.
    Returns a risk score and a final classification.
    """
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    # Convert the input data into a pandas DataFrame, as the model expects it.
    input_df = pd.DataFrame([product_features.model_dump()])

    # Make predictions using the loaded pipeline
    probability = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    # Return the result in a clear JSON format
    return {
        "prediction_label": "Underperforming" if prediction == 1 else "Not Underperforming",
        "risk_score": float(probability),
        "input_features": product_features.model_dump()
    }

