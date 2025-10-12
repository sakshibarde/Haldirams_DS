# 1. Use an official, lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first to leverage Docker's layer caching
COPY ./requirements.txt /app/requirements.txt

# 4. Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application files into the container
COPY ./best_performing_model.pkl /app/best_performing_model.pkl
COPY ./main.py /app/main.py

# 6. Expose the port that the app will run on
EXPOSE 8000

# 7. Define the command to run the application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
