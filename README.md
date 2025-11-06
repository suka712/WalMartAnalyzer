# Walmart Sales Intelligence Hub

This project provides a comprehensive solution for Walmart sales analysis and forecasting. It includes a web-based dashboard for visualizing key performance indicators (KPIs), exploring sales forecasts, simulating promotional activities, and identifying strategic insights.

## Features

*   **Executive KPI Dashboard**: A high-level overview of business health, including total sales, growth metrics, and holiday lift forecasts.
*   **Forecast Deep Dive**: Detailed sales forecasts for specific stores and departments, with the ability to analyze historical data and model residuals.
*   **Promotion Simulator**: A "what-if" tool to estimate the impact of markdown promotions on sales, providing an estimated return on investment (ROI).
*   **Strategic Insights**: Analysis of markdown ROI, store clusters, and forecast error hotspots to identify growth opportunities and operational risks.
*   **Backend API**: A Flask-based API that serves the sales prediction model.

## Project Structure

```
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── sample_request.json     # Sample request for the prediction API
├── api_backend/
│   ├── api.py              # Flask API for serving the model
│   ├── predictor.py        # Sales prediction logic
│   └── walmart_sales_model_*/ # Trained model and artifacts
└── data/
    ├── store_locations.csv # Store location data
    └── test_sample.csv     # Sample sales data
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd WalMartAnalyzer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the backend API:**
    ```bash
    python api_backend/api.py
    ```
    The API will be available at `http://127.0.0.1:5000`.

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

## API Endpoints

*   **`GET /health`**: Health check endpoint to confirm the API is running and the model is loaded.
*   **`POST /predict`**: Main prediction endpoint. Expects a JSON list of records. See `sample_request.json` for an example.
