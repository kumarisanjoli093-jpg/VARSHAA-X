/**
 * VARSHAA-X ML API Prediction Service Layer
 * Target API: http://127.0.0.1:8000
 */

const PREDICTION_API_BASE_URL = "http://127.0.0.1:8000";

const REQUIRED_FEATURES = [
  "rain_current_max_mm_h",
  "rain_current_mean_mm_h",
  "cloud_top_pressure_hpa",
  "cloud_top_temperature_k",
  "effective_cloud_emissivity"
];

class PredictionService {
  constructor(baseUrl = PREDICTION_API_BASE_URL) {
    this.baseUrl = baseUrl;
    this.isLoading = false;
    this.lastError = null;
    this.lastHealth = null;
    this.lastPrediction = null;
  }

  /**
   * 1. checkModelHealth()
   * GET http://127.0.0.1:8000/health
   */
  async checkModelHealth() {
    this.lastError = null;
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        headers: {
          "Accept": "application/json"
        }
      });

      if (!response.ok) {
        throw new Error(`API health check failed with HTTP status ${response.status}`);
      }

      const data = await response.json();

      // Validate response structure
      if (!data || typeof data !== "object" || data.status !== "ok") {
        throw new Error("Invalid health check response format from server");
      }

      this.lastHealth = data;
      return {
        success: true,
        data: data
      };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Network/API connection failure";
      this.lastError = errorMsg;
      return {
        success: false,
        error: errorMsg
      };
    }
  }

  /**
   * 2. getPrediction(features)
   * POST http://127.0.0.1:8000/predict
   */
  async getPrediction(features) {
    this.lastError = null;
    this.isLoading = true;

    try {
      // Validate input features object
      if (!features || typeof features !== "object") {
        throw new Error("Invalid input: features must be an object");
      }

      const payload = {};
      const missing = [];
      const invalidTypes = [];

      for (const featureName of REQUIRED_FEATURES) {
        if (!(featureName in features) || features[featureName] === null || features[featureName] === undefined) {
          missing.push(featureName);
          continue;
        }

        const numVal = Number(features[featureName]);
        if (isNaN(numVal)) {
          invalidTypes.push(featureName);
        } else {
          payload[featureName] = numVal;
        }
      }

      if (missing.length > 0) {
        throw new Error(`Missing required feature(s): ${missing.join(", ")}`);
      }

      if (invalidTypes.length > 0) {
        throw new Error(`Feature(s) must be valid numbers: ${invalidTypes.join(", ")}`);
      }

      // Execute POST request to ML API
      console.log("[Prediction API Request] POST http://127.0.0.1:8000/predict", payload);

      const response = await fetch(`${this.baseUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        let errDetail = `HTTP error ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson && errJson.error) errDetail = errJson.error;
        } catch (_) {}
        console.warn("[Prediction API Error]", errDetail);
        throw new Error(`Prediction request failed: ${errDetail}`);
      }

      const data = await response.json();
      console.log("[Prediction API Response] HTTP", response.status, data);

      // Validate invalid response structure
      if (!data || typeof data !== "object" || typeof data.probability !== "number" || !data.risk) {
        throw new Error("Invalid prediction response format received from ML API");
      }

      this.lastPrediction = data;
      return {
        success: true,
        data: data
      };
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Network/API connection failure";
      this.lastError = errorMsg;
      return {
        success: false,
        error: errorMsg
      };
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Helper function: Maps frontend UI metric inputs into the exact 5 ML model features
   */
  mapUIInputsToFeatures(uiInputs) {
    const rain = Number(uiInputs?.rain ?? uiInputs?.rainfall ?? 0);
    const humidity = Number(uiInputs?.humidity ?? 0);
    const cloud = Number(uiInputs?.cloud ?? uiInputs?.cloudCover ?? 0);
    const temp = Number(uiInputs?.temp ?? uiInputs?.temperature ?? 25);

    // Derived physical approximations from available environmental metrics
    const rainMax = Math.max(0, rain);
    const rainMean = Math.max(0, rain * 0.6);
    // Pressure drops from standard sea level pressure (1013 hPa) based on cloud cover depth
    const cloudPressure = Math.max(100, Math.min(1013.25, 1013.25 - (cloud / 100) * 600));
    // Cloud top temperature in Kelvin
    const cloudTempK = Math.max(180, Math.min(320, (temp + 273.15) - (cloud / 100) * 40));
    // Effective emissivity scale [0.05 - 1.0]
    const cloudEmissivity = Math.max(0.05, Math.min(1.0, cloud / 100));

    return {
      rain_current_max_mm_h: rainMax,
      rain_current_mean_mm_h: rainMean,
      cloud_top_pressure_hpa: cloudPressure,
      cloud_top_temperature_k: cloudTempK,
      effective_cloud_emissivity: cloudEmissivity
    };
  }
}

// Global instance export for browser script inclusion
const predictionService = new PredictionService();

if (typeof window !== "undefined") {
  window.PredictionService = PredictionService;
  window.predictionService = predictionService;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { PredictionService, predictionService };
}
