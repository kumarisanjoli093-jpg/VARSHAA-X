# VARSHAA-X XAI Prediction API

This replaces the previous prototype API.

## What changed

- Removes the visible sklearn 1.8 -> 1.9 persistence warning.
- Adds `/` route.
- Keeps `/health`.
- `/predict` now returns:
  - probability
  - risk
  - target
  - top 3 local feature explanations
  - detailed feature perturbation effects

The explanation method is deliberately described as **feature perturbation**. It is not presented as a causal explanation or calibrated probability contribution.

## Install

From:

D:\VARSHAA-X\VARSHAA-X\varshaa_x_prediction_api

run:

    python -m pip install -r varshaa_xai_api/requirements.txt

If the old Flask server is still running, press Ctrl+C first.

## Run

    python varshaa_xai_api/app.py

Then test:

    Invoke-RestMethod http://127.0.0.1:8000/health

## Prediction test

    $body = @{
      rain_current_max_mm_h = 18.4
      rain_current_mean_mm_h = 7.2
      cloud_top_pressure_hpa = 610
      cloud_top_temperature_k = 242
      effective_cloud_emissivity = 0.91
    } | ConvertTo-Json

    Invoke-RestMethod `
      -Uri http://127.0.0.1:8000/predict `
      -Method POST `
      -ContentType "application/json" `
      -Body $body

## Prototype limitation

The model was trained on a limited temporal satellite sample. The returned
score is a prototype model score, not a calibrated nationwide probability.
The explanation is a local perturbation explanation, not a causal claim.
