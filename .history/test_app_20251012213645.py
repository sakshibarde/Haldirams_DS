import requests
from fastapi.testclient import TestClient
from main import app  # Import the FastAPI app from your main.py file

# Create a test client for the FastAPI app
client = TestClient(app)

def test_predict_underperforming():
    """
    Test the /predict endpoint with data that should result in an 'Underperforming' prediction.
    """
    test_data = {
        "rating": 3.5,
        "price_whole": 250,
        "mrp": 250,
        "number_of_global_ratings": 15,
        "number_of_reviews": 10,
        "discount_percentage_cleaned": 0,
        "product_weight_grams": 400,
        "category_reclassified": "Spicy Snacks"
    }
    response = client.post("/predict", json=test_data)
    
    # Assert that the request was successful
    assert response.status_code == 200
    
    # Assert that the prediction is correct
    json_response = response.json()
    assert json_response["prediction_label"] == "Underperforming"
    assert "risk_score" in json_response

def test_predict_not_underperforming():
    """
    Test the /predict endpoint with data for a product that should perform well.
    """
    test_data = {
        "rating": 4.5,
        "price_whole": 150,
        "mrp": 150,
        "number_of_global_ratings": 5000,
        "number_of_reviews": 2500,
        "discount_percentage_cleaned": 0,
        "product_weight_grams": 200,
        "category_reclassified": "Namkeen"
    }
    response = client.post("/predict", json=test_data)
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["prediction_label"] == "Not Underperforming"
    assert json_response["risk_score"] < 0.5