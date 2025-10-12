import requests
import json

# The URL of your locally running API inside the Docker container
API_URL = "http://127.0.0.1:8000/predict"

# Sample data for a new product that is likely to be an underperformer
new_product_data = {
    "rating": 3.5,
    "price_whole": 250,
    "mrp": 250,
    "number_of_global_ratings": 15,
    "number_of_reviews": 10,
    "discount_percentage_cleaned": 0,
    "product_weight_grams": 400,
    "category_reclassified": "Spicy Snacks"
}

print("--- Sending request to the API ---")
print("Data:", json.dumps(new_product_data, indent=2))

try:
    # Send a POST request to the API
    response = requests.post(API_URL, json=new_product_data)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Print the JSON response from the API
    print("\n--- API Response ---")
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred: {e}")
    print("Is the Docker container running?")


### **Step 3: Build, Run, and Test in VS Code Terminal**

# Open the integrated terminal in VS Code (`Ctrl` + `~` or `View -> Terminal`).

# 1.  **Build the Docker Image:** This command reads your `Dockerfile` and builds a self-contained image named `haldirams-api`.
#     ```bash
#     docker build -t haldirams-api .
#     ```

# 2.  **Run the Docker Container:** This command starts a container from your image, runs it in the background (`-d`), and connects your computer's port 8000 to the container's port 8000 (`-p`).
#     ```bash
#     docker run -d -p 8000:8000 --name haldirams-container haldirams-api
#     ```

# 3.  **Test the Running API:** Now that your API is live in the container, run the test script.
#     ```bash
#     python test_api.py