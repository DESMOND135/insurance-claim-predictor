# Insurance Claim Prediction App

This project represents an industrial-level Machine Learning application that predicts insurance claim amounts based on several personal factors such as age, BMI, smoker status, and health history. 

## Project Architecture

The architecture includes:
- **Machine Learning Components (`src/`)**: A modularized ML pipeline for reading data, preprocessing (using `StandardScaler` and `OneHotEncoder`), and training a `RandomForestRegressor`. It includes custom exception handling and logging.
- **FastAPI Backend (`app.py`)**: A backend service that exposes a `/predict` REST API endpoint to serve model inferences.
- **Streamlit Frontend (`streamlit_app.py`)**: A modern, interactive user interface allowing users to input their data and view predictions.
- **Dockerization (`Dockerfile`)**: A containerized setup allowing both the frontend and backend to run simultaneously within a single Docker container.

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ and Docker installed.

### 1. Local Setup

First, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Train the model to generate the `.pkl` files (this reads `insurance.csv`):

```bash
python train.py
```

Start the FastAPI Backend:

```bash
python -m uvicorn app:app --port 8000
```

In a separate terminal, start the Streamlit Frontend:

```bash
streamlit run streamlit_app.py --server.port 8501
```

Access the Streamlit app at `http://localhost:8501` and FastAPI docs at `http://localhost:8000/docs`.

### 2. Docker Setup

You can build and run the entire application using Docker:

```bash
docker build -t insurance_app .
docker run -p 8000:8000 -p 8501:8501 insurance_app
```

Then visit `http://localhost:8501` in your browser.

## Git Workflow Simulation

This repository follows a standard industrial Git Workflow:
1. The **`main`** branch contains production-ready code.
2. Development occurs on the **`work-branch`**.
3. Once a feature is complete, a Pull Request (PR) is opened from `work-branch` to `main`.
