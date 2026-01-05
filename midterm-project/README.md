# 🏡 Automated Real Estate Valuation Model (AVM)

## 1. Problem Description and Business Goal

The objective of this project is to build an **Automated Valuation Model (AVM)** for the real estate market in Poland, specifically focusing on the Warsaw metropolitan area.

The core business problem is the **inefficiency of identifying undervalued assets**. 
In the 2025 housing market, private buyers often rely on human intuition and cannot efficiently assess the fair price of a property. Furthermore, the volume of offers buyers need to analyze daily is enormous. With dynamic pricing, it is extremely important to have an automated service that directs a buyer's attention to specific, potentially undervalued offers.

The solution aims to provide a reliable, objective, and near real-time "Fair Value" estimate for any given property listing, allowing users to **identify undervalued assets** instantly.


### 📂 Dataset

The dataset used for this project is **included in the repository**:
* File: `mazowieckie-spring25.csv`
* Source: Scraped data from major Polish real estate listing portals (Spring 2025 snapshot).

---

## 2. EDA

## 3. Machine Learning Formulation, Model training

This project is framed as a **Supervised Regression** problem:

* **Input (Features - X):** Structural, locational, and amenity-based characteristics (e.g., `area`, `buildYear`, `floor_numeric`, `distance_from_center`) and engineered boolean features (e.g., `has_elevator`, `has_terrace`).
* **Output (Target - y):** The logarithm of the property's asking price (`price_log`).
* **Model:** An optimized **Extreme Gradient Boosting (XGBoost)** Regressor, selected for its effectiveness in capturing non-linear feature interactions common in real estate data.

The model was developed using a dataset of **20,000+ flat listings** involving extensive cleaning, outlier removal, and feature engineering:

* **Target Transformation:** Using $\ln(1 + \text{price})$ to stabilize the price distribution, training the model to minimize **percentage error** rather than absolute error.
* **High-Cardinality Encoding:** Employing **Target Encoding** for the high-cardinality `location_district` feature to efficiently capture the value hierarchy of different city zones.

---

## 4. Dependency and Environment Management

This project uses **[uv](https://github.com/astral-sh/uv)** for fast dependency management and virtual environment creation.

### Installation & Setup
To reproduce the environment locally, follow these steps:

1.  **Install uv** (if not already installed):
    ```bash
    pip install uv
    ```

2.  **Sync Dependencies**:
    This command will create a virtual environment (`.venv`) and install all locked dependencies from `uv.lock`.
    ```bash
    uv sync
    ```

3.  **Activate the Environment**:
    * **Mac/Linux**:
        ```bash
        source .venv/bin/activate
        ```
    * **Windows**:
        ```bash
        .venv\Scripts\activate
        ```

---

## 5.🔄 Reproducibility

To reproduce the findings and model training process:

### 1. Re-train the Model (Production Pipeline)
The core training logic is exported to `train.py`, which utilizes data transformations defined in `transform.py`.

To train the final model and generate the `model_pipeline.bin` file:

```bash
# Ensure you are in the project root and the environment is active
python train.py
```

* Input: Reads mazowieckie-spring25.csv.

* Process: Cleans data and engineers features (via transform.py), trains the XGBoost model.

* Output: Saves the trained artifact to model_pipeline.bin.

### 2. Explore the Analysis (Optional)

If you want to inspect the Exploratory Data Analysis (EDA) and the hyperparameter tuning process:

* Open EDA.ipynb in your preferred notebook editor
* Run the cells to visualize the data distribution, feature importance, and initial model experiments.

## 6.🐋 Containerization (Local Deployment)

The application is containerized using Docker. The image includes the **FastAPI** prediction service (`predict.py`) and the trained model.

### 1. Build the Docker Image
Run the following command in the project root:

```bash
docker build -t mid-term-project .
```
### 2. Run the Container
Run the container, mapping port 9696 on your host to port 9696 in the container:

```bash
docker run -it --rm -p 9696:9696 mid-term-project
```

### 3. Test the Local Deployment

FastAPI automatically provides Swagger UI, which is very helpful.
API Docs are available at http://localhost:9696/docs

* Navigate to http://localhost:9696/docs
* Click POST /predict -> Try it out.
* Paste your JSON data (ex. test_record.json content) and click Execute.

## 7. ☁️ Cloud Deployment (Fly.io)

This project is deployed to the cloud using Fly.io.

### 1. Prerequisites

* Install flyctl
* Login: fly auth login

### 2. Launch the app

```bash
fly launch --no-deploy
```

* Name: Choose a unique name (e.g., waw-flat-pricer).
* Region: Choose a region close to you (e.g., waw for Warsaw).

### 3. Configure Port

Edit the generated fly.toml file to ensure the internal port matches our FastAPI app (9696).

Find the [http_service] section and set internal_port:

```toml
[http_service]
  internal_port = 9696
  force_https = true
  auto_stop_machines = 'stop'
  auto_start_machines = true
  min_machines_running = 0
  processes = ['app']
```

### 4. Deploy the app
```bash
fly deploy
```

### 5. Access the Cloud Service
Once deployed, your AVM is accessible globally:

* API Docs: https://<app_name>.fly.dev/docs
* Prediction Endpoint: https://<app_name>.fly.dev/predict