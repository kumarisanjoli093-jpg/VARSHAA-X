/**
 * VARSHAA-X Local MOSDAC Feature Provider Layer
 * Target Data Source: data/raw/HEM & data/raw/CTP (INSAT-3DR L2B MOSDAC Datasets)
 * Timestamp: 19-AUG-2026 04:15 UTC (19AUG2026_0415)
 * 
 * Responsibilities:
 * - Accepts global location object: { name, latitude, longitude }
 * - Maps latitude/longitude to the nearest valid MOSDAC satellite grid cell
 * - Returns the exact 5 real model features or an explicit unavailable error state
 * 
 * Architecture Flow:
 * Global Location -> featureProvider.getModelFeatures() -> 5 Real MOSDAC Features -> predictionService.getPrediction() -> Prediction + XAI -> UI
 */

const MOSDAC_DATA_URL = "data/mosdac_features.json";
const MAX_SEARCH_RADIUS_DEG = 1.5; // Maximum spatial search distance (~150km)

class FeatureProvider {
  constructor() {
    this.mosdacData = null;
    this.loadPromise = null;
    this.isLoaded = false;
  }

  /**
   * Load MOSDAC dataset index
   */
  async loadMosdacData() {
    if (this.isLoaded && this.mosdacData) return this.mosdacData;
    if (this.loadPromise) return this.loadPromise;

    this.loadPromise = (async () => {
      try {
        if (typeof window === "undefined" && typeof process !== "undefined") {
          // Node environment support for tests
          const fs = require("fs");
          const path = require("path");
          const p = path.resolve(__dirname, "../data/mosdac_features.json");
          if (fs.existsSync(p)) {
            this.mosdacData = JSON.parse(fs.readFileSync(p, "utf8"));
            this.isLoaded = true;
            return this.mosdacData;
          }
        }

        const res = await fetch(MOSDAC_DATA_URL);
        if (!res.ok) {
          throw new Error(`Failed to load MOSDAC satellite dataset: HTTP ${res.status}`);
        }
        this.mosdacData = await res.json();
        this.isLoaded = true;
        return this.mosdacData;
      } catch (err) {
        console.warn("FeatureProvider MOSDAC load notice:", err);
        this.mosdacData = null;
        return null;
      } finally {
        this.loadPromise = null;
      }
    })();

    return this.loadPromise;
  }

  /**
   * getModelFeatures(location)
   * Retrieves real MOSDAC satellite feature values for a given global location.
   * 
   * @param {Object} location - { name, latitude, longitude }
   * @returns {Promise<Object>} Object containing 5 MOSDAC features or unavailable state
   */
  async getModelFeatures(location) {
    if (!location) {
      return {
        available: false,
        reason: "Satellite data unavailable for this location/time."
      };
    }

    const lat = Number(location.latitude ?? location.lat);
    const lng = Number(location.longitude ?? location.lng);

    if (isNaN(lat) || isNaN(lng)) {
      return {
        available: false,
        reason: "Satellite data unavailable for this location/time."
      };
    }

    await this.loadMosdacData();

    if (!this.mosdacData || !this.mosdacData.grid) {
      return {
        available: false,
        reason: "Satellite data unavailable for this location/time."
      };
    }

    const grid = this.mosdacData.grid;
    const latGrid = Math.round(lat * 4) / 4;
    const lonGrid = Math.round(lng * 4) / 4;
    const directKey = `${latGrid.toFixed(2)},${lonGrid.toFixed(2)}`;

    let bestMatch = null;
    let minDistance = Infinity;
    let bestKey = null;

    if (grid[directKey]) {
      bestMatch = grid[directKey];
      bestKey = directKey;
      minDistance = 0;
    } else {
      for (const key in grid) {
        const parts = key.split(",");
        if (parts.length !== 2) continue;
        const keyLat = parseFloat(parts[0]);
        const keyLng = parseFloat(parts[1]);

        const dLat = lat - keyLat;
        const dLng = lng - keyLng;
        const dist = Math.sqrt(dLat * dLat + dLng * dLng);

        if (dist < minDistance && dist <= MAX_SEARCH_RADIUS_DEG) {
          minDistance = dist;
          bestMatch = grid[key];
          bestKey = key;
        }
      }
    }

    if (!bestMatch) {
      return {
        available: false,
        reason: "Satellite data unavailable for this location/time."
      };
    }

    // Extract exact 5 features
    const features = {
      rain_current_max_mm_h: Number(bestMatch.rain_current_max_mm_h || 0),
      rain_current_mean_mm_h: Number(bestMatch.rain_current_mean_mm_h || 0),
      cloud_top_pressure_hpa: Number(bestMatch.cloud_top_pressure_hpa || 0),
      cloud_top_temperature_k: Number(bestMatch.cloud_top_temperature_k || 0),
      effective_cloud_emissivity: Number(bestMatch.effective_cloud_emissivity || 0)
    };

    // Development Debug Logging (Requirement 9)
    console.log(`[Satellite Feature Provider]`);
    console.log(`Location: ${location.name || location.label || "Selected Location"} (${lat.toFixed(4)}, ${lng.toFixed(4)})`);
    console.log(`Timestamp: ${this.mosdacData.timestamp_formatted || this.mosdacData.timestamp}`);
    console.log(`Nearest grid point: HEM [${bestMatch.nearest_grid_hem}], CTP [${bestMatch.nearest_grid_ctp}]`);
    console.log(`Distance: HEM ${bestMatch.dist_hem_deg}°, CTP ${bestMatch.dist_ctp_deg}°`);
    console.log(`HEM rainfall max: ${features.rain_current_max_mm_h} mm/h`);
    console.log(`HEM rainfall mean: ${features.rain_current_mean_mm_h} mm/h`);
    console.log(`CTP: ${features.cloud_top_pressure_hpa} hPa`);
    console.log(`CTT: ${features.cloud_top_temperature_k} K`);
    console.log(`Emissivity: ${features.effective_cloud_emissivity}`);

    return {
      available: true,
      timestamp: this.mosdacData.timestamp,
      timestamp_formatted: this.mosdacData.timestamp_formatted,
      nearest_grid_hem: bestMatch.nearest_grid_hem,
      nearest_grid_ctp: bestMatch.nearest_grid_ctp,
      distance_hem_deg: bestMatch.dist_hem_deg,
      distance_ctp_deg: bestMatch.dist_ctp_deg,
      features: features,
      nowcast: bestMatch.nowcast || null,
      // Backward compatibility for existing callers
      rain_current_max_mm_h: features.rain_current_max_mm_h,
      rain_current_mean_mm_h: features.rain_current_mean_mm_h,
      cloud_top_pressure_hpa: features.cloud_top_pressure_hpa,
      cloud_top_temperature_k: features.cloud_top_temperature_k,
      effective_cloud_emissivity: features.effective_cloud_emissivity
    };
  }
}

// Global instance export
const featureProvider = new FeatureProvider();

if (typeof window !== "undefined") {
  window.FeatureProvider = FeatureProvider;
  window.featureProvider = featureProvider;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { FeatureProvider, featureProvider };
}
