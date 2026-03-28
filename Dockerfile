FROM python:3.10-slim

WORKDIR /app

# Copy all files to the container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the training script to generate the model and preprocessor inside the container
# Alternately it could have been mapped if generated locally
RUN python train.py

# Give execution rights on the start script
RUN chmod +x start.sh

# Expose ports for FastAPI(8000) and Streamlit(8501)
EXPOSE 8000
EXPOSE 8501

# Command to run both the FastAPI server and Streamlit UI
CMD ["./start.sh"]
