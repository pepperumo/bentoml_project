# Admissions Prediction Service

A BentoML-based service that predicts the chances of admission to graduate school based on student qualifications.

## Project Structure

```
bentoml_project/
├── bentoml-env/         # Virtual environment (ignored by Git)
├── data/
│   ├── processed/       # Will contain processed datasets
│   └── raw/             # Contains raw admission.csv
├── models/              # Will store trained models
├── src/
│   ├── prepare_data.py  # Data preparation script
│   ├── train_model.py   # Model training script
│   └── service.py       # BentoML service with API endpoints
├── tests/
│   └── test_service.py  # Unit tests for the API service
├── bentofile.yaml       # BentoML configuration
└── README.md            # This file
```

## Setup

### 1. Decompress the Archive

If the project is delivered as an archive (ZIP or tar.gz), decompress it using:

- **For tar.gz:**
    ```bash
    tar -xvf project_archive.tar.gz
    cd bentoml_project
    ```

- **For ZIP:**
    ```bash
    unzip project_archive.zip
    cd bentoml_project
    ```

### 2. Virtual Environment

Create and activate your virtual environment:

#### Windows
```bash
python -m venv bentoml-env
bentoml-env\Scripts\activate
```

#### Mac/Linux
```bash
python -m venv bentoml-env
source bentoml-env/bin/activate
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install bentoml scikit-learn pandas numpy PyJWT pytest requests
```

## Running the Project

### 4. Prepare Data

Run the data preparation script to process the raw admission data:

```bash
python src/prepare_data.py
```

This will:
- Load data from `data/raw/admission.csv`
- Clean and preprocess the data
- Split it into training and testing sets
- Save the processed datasets to `data/processed/`

### 5. Train the Model

Train a regression model using the processed data:

```bash
python src/train_model.py
```

This will:
- Load the processed datasets
- Train and evaluate a regression model
- Save the model to the BentoML model store (if performance is acceptable)

Verify that your model was saved:

```bash
bentoml models list
```

### 6. Run the Service Locally

For development, serve your BentoML service:

```bash
bentoml serve src.service:svc --reload
```

The API will be available at http://localhost:3000

### 7. Test the API

Run the unit tests to verify the API functionality:

```bash
pytest tests/test_service.py -v
```

All tests should pass with the expected HTTP status codes.

## API Usage Examples

### Authentication

Obtain a JWT token by sending a login request:

```bash
curl -X POST http://localhost:3000/login \
    -H "Content-Type: application/json" \
    -d '{"username": "admin", "password": "admin123"}'
```

Response:
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
}
```

### Making Predictions

Use the token to authenticate prediction requests:

```bash
curl -X POST http://localhost:3000/predict \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_TOKEN_HERE" \
    -d '{
        "GRE_Score": 337,
        "TOEFL_Score": 118,
        "University_Rating": 4,
        "SOP": 4.5,
        "LOR": 4.5,
        "CGPA": 9.65,
        "Research": 1
    }'
```

Response:
```json
{
    "chance_of_admit": 0.92
}
```

## Building and Containerizing

### 8. Build a Bento

Package your service as a Bento:

```bash
bentoml build
```

### 9. Containerize the Bento

Create a Docker container from your Bento with the desired tag:

```bash
bentoml containerize admissions_prediction:latest --image-tag your_name_admissions_prediction:latest
```

This command builds a Docker image named `your_name_admissions_prediction:latest`.

### 10. Run the Docker Container

Run the container locally to serve the API on port 3000:

```bash
docker run --rm -p 3000:3000 your_name_admissions_prediction:latest
```

### 11. Export the Docker Image

Export the Docker image as a tarball:

```bash
docker save -o bento_image.tar your_name_admissions_prediction:latest
```

## Security Notes

For production, consider:

1. Replacing the hardcoded `SECRET_KEY` in `src/service.py` with a secure environment variable
2. Implementing a robust user authentication system or integrating with an identity provider
3. Adding rate limiting and other security measures
4. Using HTTPS for all API communications

## Final Workflow Recap

1. **Decompress the Archive:**
     ```
     tar -xvf project_archive.tar.gz or unzip project_archive.zip
     cd bentoml_project
     ```

2. **Activate the Virtual Environment:**
     ```
     python -m venv bentoml-env
     bentoml-env\Scripts\activate (Windows) or source bentoml-env/bin/activate (Mac/Linux)
     ```

3. **Install Dependencies:**
     ```
     pip install bentoml scikit-learn pandas numpy PyJWT pytest requests
     ```

4. **Prepare Data & Train Model:**
     ```
     python src/prepare_data.py
     python src/train_model.py
     ```
     Verify with `bentoml models list`

5. **Run the Service Locally:**
     ```
     bentoml serve src.service:svc --reload
     ```

6. **Run Unit Tests:**
     ```
     pytest tests/test_service.py -v
     ```

7. **Build & Containerize the Bento:**
     ```
     bentoml build
     bentoml containerize admissions_prediction:latest --image-tag your_name_admissions_prediction:latest
     ```

8. **Export the Docker Image:**
     ```
     docker save -o bento_image.tar your_name_admissions_prediction:latest
     ```

9. **Run the Docker Container:**
     ```
     docker run --rm -p 3000:3000 your_name_admissions_prediction:latest
     ```
