# VARSHAA-X Prediction API

This is a small local Flask API around the trained VARSHAA-X Random Forest model.

## Folder structure

Keep this structure:

VARSHAA-X/
├── varshaa_x_api/
│   ├── app.py
│   └── requirements.txt
└── varshaa_x_model/
    └── varshaa_x_random_forest.pkl

## Install

From the VARSHAA-X project root:

    python -m pip install -r varshaa_x_api/requirements.txt

## Run

    python varshaa_x_api/app.py

The API runs at:

    http://127.0.0.1:8000

## Test health

Open:

    http://127.0.0.1:8000/health

## Test prediction

PowerShell:

    $body = @{
      rain_current_max_mm_h = 18.4
      rain_current_mean_mm_h = 7.2
      cloud_top_pressure_hpa = 610
      cloud_top_temperature_k = 242
      effective_cloud_emissivity = 0.91
    } | ConvertTo-Json

    Invoke-RestMethod -Uri http://127.0.0.1:8000/predict `
      -Method POST `
      -ContentType "application/json" `
      -Body $body

The response contains the model probability and a LOW/MODERATE/HIGH prototype risk band.

Important:
This API exposes the trained prototype model. Its metrics were obtained from a limited temporal satellite sample, so the probability should be presented as a prototype model score, not a calibrated nationwide probability.
