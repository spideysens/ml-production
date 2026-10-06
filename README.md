# Amazon Review Sentiment API

This project trains a machine-learning model that predicts whether an Amazon
review has **positive** or **negative** sentiment. It demonstrates the complete
workflow from training and exporting a model to serving predictions through a
FastAPI application and deploying the application as a Docker container.

For the complete beginner-friendly workflow, download and open
[Amazon_Review_Sentiment_API_Render_Tutorial.docx](Amazon_Review_Sentiment_API_Render_Tutorial.docx).

## Project structure

```text
.
|-- datasets/
|   `-- amazon_dataset.csv
|-- train.py
|-- main.py
|-- model.pkl
|-- requirements.txt
|-- Dockerfile
|-- Amazon_Review_Sentiment_API_Render_Tutorial.docx
`-- README.md
```

## API endpoints

### `GET /health`

Reports whether the API is healthy and the trained model was loaded.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Example request body:

```json
{
  "review": "This product is excellent and works perfectly."
}
```

Example response:

```json
{
  "prediction": 1,
  "sentiment": "positive",
  "confidence": 0.9732
}
```

`prediction` is `1` for positive sentiment and `0` for negative sentiment.

## Run locally

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train and export the model:

```powershell
python train.py
```

Start the API:

```powershell
uvicorn main:app --reload
```

Open the interactive documentation at <http://127.0.0.1:8000/docs>.

## Run with Docker

```powershell
docker build -t amazon-sentiment-api .
docker run --rm -p 8000:10000 amazon-sentiment-api
```

Then open <http://127.0.0.1:8000/docs>.

## Test from PowerShell

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{
    review = "This product is excellent and easy to use."
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## Deployment

The included `Dockerfile` is ready for deployment as a Render Web Service.
Set the Render health-check path to `/health`. The application listens on the
port supplied through Render's `PORT` environment variable.
