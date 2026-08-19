"""
VARSHAA-X optimized HEM + CTP pipeline.

This version avoids the extremely expensive per-pixel KD-tree query used in
the earlier prototype. For the regular INSAT grids, it computes the
HEM->CTP mapping once from the latitude/longitude axes and reuses it.

Run from the VARSHAA-X project root:
    python varshaa_x_pipeline_v4.py --hem data/raw/HEM --ctp data/raw/CTP \
      --out data/datasets/varshaa_x_training.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

THRESHOLD_MM_H = 50.0
STEP_MINUTES = 30
FUTURE_STEPS = 6  # 3 hours at 30-minute intervals
FILL = -999.0


def acquisition_key(path: Path):
    m = re.search(r"_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", path.name.upper())
    if not m:
        return None
    return pd.to_datetime(
        m.group(1) + m.group(2), format="%d%b%Y%H%M", utc=True
    )


def normalize_grid(arr, lat):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and lat.ndim == 2 and arr.size == lat.size:
        arr = arr.reshape(lat.shape)
    if arr.shape != lat.shape:
        raise ValueError(f"Array shape {arr.shape} != coordinate shape {lat.shape}")
    return arr


def read_hem(path: Path):
    with h5py.File(path, "r") as h:
        lat_ds = h["Latitude"]
        lon_ds = h["Longitude"]
        lat_raw = np.asarray(lat_ds[:], dtype=np.float32)
        lon_raw = np.asarray(lon_ds[:], dtype=np.float32)

        # MOSDAC stores these coordinates as packed int16 values. Apply the
        # dataset's CF scale/offset and explicitly mask its fill value.
        lat_fill = float(np.asarray(lat_ds.attrs.get("_FillValue", [32767])).ravel()[0])
        lon_fill = float(np.asarray(lon_ds.attrs.get("_FillValue", [32767])).ravel()[0])
        lat_scale = float(np.asarray(lat_ds.attrs.get("scale_factor", [1.0])).ravel()[0])
        lon_scale = float(np.asarray(lon_ds.attrs.get("scale_factor", [1.0])).ravel()[0])
        lat_offset = float(np.asarray(lat_ds.attrs.get("add_offset", [0.0])).ravel()[0])
        lon_offset = float(np.asarray(lon_ds.attrs.get("add_offset", [0.0])).ravel()[0])

        lat = np.where(lat_raw == lat_fill, np.nan, lat_raw * lat_scale + lat_offset)
        lon = np.where(lon_raw == lon_fill, np.nan, lon_raw * lon_scale + lon_offset)

        rain = normalize_grid(h["HEM"][:], lat)

    rain[rain <= FILL] = np.nan
    return rain, lat, lon


def read_ctp(path: Path):
    with h5py.File(path, "r") as h:
        lat_ds = h["Latitude"]
        lon_ds = h["Longitude"]
        lat_raw = np.asarray(lat_ds[:], dtype=np.float32)
        lon_raw = np.asarray(lon_ds[:], dtype=np.float32)

        lat_fill = float(np.asarray(lat_ds.attrs.get("_FillValue", [32767])).ravel()[0])
        lon_fill = float(np.asarray(lon_ds.attrs.get("_FillValue", [32767])).ravel()[0])
        lat_scale = float(np.asarray(lat_ds.attrs.get("scale_factor", [1.0])).ravel()[0])
        lon_scale = float(np.asarray(lon_ds.attrs.get("scale_factor", [1.0])).ravel()[0])
        lat_offset = float(np.asarray(lat_ds.attrs.get("add_offset", [0.0])).ravel()[0])
        lon_offset = float(np.asarray(lon_ds.attrs.get("add_offset", [0.0])).ravel()[0])

        lat = np.where(lat_raw == lat_fill, np.nan, lat_raw * lat_scale + lat_offset)
        lon = np.where(lon_raw == lon_fill, np.nan, lon_raw * lon_scale + lon_offset)

        ctp = normalize_grid(h["CTP"][:], lat)
        ctt = normalize_grid(h["CTT"][:], lat)
        emiss = normalize_grid(h["EFF_EMISS"][:], lat)

    for a in (ctp, ctt, emiss):
        a[a <= FILL] = np.nan
    return ctp, ctt, emiss, lat, lon


def axis_from_grid(lat, lon):
    """Extract rectilinear latitude/longitude axes."""
    if lat.ndim == 1 and lon.ndim == 1:
        return lat, lon

    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("Expected 1-D or 2-D latitude/longitude grids.")

    lat_axis = lat[:, 0]
    lon_axis = lon[0, :]

    # Verify approximately rectilinear.
    if not (
        np.nanmax(np.abs(lat - lat_axis[:, None])) < 1e-3
        and np.nanmax(np.abs(lon - lon_axis[None, :])) < 1e-3
    ):
        raise ValueError("Grid is not rectilinear; fallback spatial mapping is required.")

    return lat_axis, lon_axis


def nearest_indices(values, axis):
    """Nearest index on a monotonic axis, preserving descending axes."""
    axis = np.asarray(axis)
    values = np.asarray(values)

    if axis[0] > axis[-1]:
        rev = axis[::-1]
        idx = np.searchsorted(rev, values)
        idx = np.clip(idx, 0, len(rev) - 1)
        left = np.clip(idx - 1, 0, len(rev) - 1)
        choose_left = np.abs(values - rev[left]) <= np.abs(values - rev[idx])
        out = np.where(choose_left, left, idx)
        return len(axis) - 1 - out

    idx = np.searchsorted(axis, values)
    idx = np.clip(idx, 0, len(axis) - 1)
    left = np.clip(idx - 1, 0, len(axis) - 1)
    choose_left = np.abs(values - axis[left]) <= np.abs(values - axis[idx])
    return np.where(choose_left, left, idx)


class SpatialMapper:
    """Build the HEM -> CTP nearest-neighbour mapping once.

    INSAT grids can be geostationary/curvilinear, so we must not assume that
    latitude/longitude form a simple rectangular axis. The expensive KD-tree
    construction is performed only once; every later timestamp reuses the
    resulting HEM-pixel -> CTP-cell index mapping.
    """
    def __init__(self, hem_lat, hem_lon, ctp_lat, ctp_lon):
        hlat = np.asarray(hem_lat, dtype=np.float32)
        hlon = np.asarray(hem_lon, dtype=np.float32)
        clat = np.asarray(ctp_lat, dtype=np.float32)
        clon = np.asarray(ctp_lon, dtype=np.float32)

        if hlat.shape != hlon.shape:
            raise ValueError(f"HEM lat/lon shapes differ: {hlat.shape} vs {hlon.shape}")
        if clat.shape != clon.shape:
            raise ValueError(f"CTP lat/lon shapes differ: {clat.shape} vs {clon.shape}")

        self.hem_shape = hlat.shape
        self.ctp_shape = clat.shape

        ctp_valid = np.isfinite(clat) & np.isfinite(clon)
        if not np.any(ctp_valid):
            raise ValueError("No valid CTP latitude/longitude points found.")

        ctp_points = np.c_[clat[ctp_valid].ravel(), clon[ctp_valid].ravel()]
        self.ctp_flat_indices = np.flatnonzero(ctp_valid.ravel()).astype(np.int32)

        print("  Building one KD-tree for the CTP grid...")
        tree = cKDTree(ctp_points)

        hem_valid = np.isfinite(hlat) & np.isfinite(hlon)
        self.valid_hem_flat = np.flatnonzero(hem_valid.ravel()).astype(np.int32)

        # Query all HEM coordinates once. This is the expensive step, but it
        # happens only once for the entire run.
        print(f"  Mapping {self.valid_hem_flat.size:,} valid HEM pixels...")
        hem_points = np.c_[hlat.ravel()[self.valid_hem_flat],
                           hlon.ravel()[self.valid_hem_flat]]
        _, nearest = tree.query(hem_points, k=1, workers=-1)

        self.ctp_index_for_valid_hem = self.ctp_flat_indices[nearest].astype(np.int32)
        self.n_ctp = clat.size

    def aggregate(self, rain):
        vals_all = np.asarray(rain, dtype=np.float32).ravel()
        vals = vals_all[self.valid_hem_flat]

        valid = np.isfinite(vals)
        idx = self.ctp_index_for_valid_hem[valid]
        vals = vals[valid]

        max_r = np.full(self.n_ctp, np.nan, dtype=np.float32)
        mean_r = np.full(self.n_ctp, np.nan, dtype=np.float32)
        counts = np.bincount(idx, minlength=self.n_ctp).astype(np.int32)

        if len(vals):
            order = np.argsort(idx)
            idx_s = idx[order]
            val_s = vals[order]
            starts = np.r_[0, np.flatnonzero(np.diff(idx_s)) + 1]
            ends = np.r_[starts[1:], len(idx_s)]

            for a, b in zip(starts, ends):
                cell = idx_s[a]
                segment = val_s[a:b]
                max_r[cell] = np.nanmax(segment)
                mean_r[cell] = np.nanmean(segment)

        return max_r, mean_r, counts


def build_feature_frame(t, hem_path, ctp_path, mapper=None):
    rain, hlat, hlon = read_hem(hem_path)
    ctp, ctt, emiss, clat, clon = read_ctp(ctp_path)

    if mapper is None:
        mapper = SpatialMapper(hlat, hlon, clat, clon)

    rain_max, rain_mean, counts = mapper.aggregate(rain)

    return pd.DataFrame({
        "timestamp": t,
        "latitude": clat.ravel(),
        "longitude": clon.ravel(),
        "rain_current_max_mm_h": rain_max,
        "rain_current_mean_mm_h": rain_mean,
        "cloud_top_pressure_hpa": ctp.ravel(),
        "cloud_top_temperature_k": ctt.ravel(),
        "effective_cloud_emissivity": emiss.ravel(),
        "hem_pixels_mapped": counts,
    }), mapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hem", required=True)
    ap.add_argument("--ctp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hem_dir = Path(args.hem)
    ctp_dir = Path(args.ctp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hems = {
        acquisition_key(p): p
        for p in hem_dir.glob("*.h5")
        if acquisition_key(p) is not None
    }
    ctps = {
        acquisition_key(p): p
        for p in ctp_dir.glob("*.h5")
        if acquisition_key(p) is not None
    }

    common = sorted(set(hems) & set(ctps))
    if not common:
        raise SystemExit("No matching HEM/CTP acquisition timestamps found.")

    print(f"Matched HEM/CTP timestamps: {len(common)}")
    print("Building spatial mapping once...")

    # Build mapper from the first common pair.
    # The HEM/CTP grids are curvilinear, so SpatialMapper uses a one-time
    # KD-tree instead of assuming a rectilinear grid.
    _, mapper = build_feature_frame(
        common[0], hems[common[0]], ctps[common[0]], mapper=None
    )
    print(
        f"HEM grid: {mapper.hem_shape[0]} x {mapper.hem_shape[1]} | "
        f"CTP grid: {mapper.ctp_shape[0]} x {mapper.ctp_shape[1]}"
    )

    # Read/aggregate all available HEM timestamps once.
    # For a one-day test this is small; avoids repeated high-resolution work.
    print("Aggregating HEM rainfall onto CTP grid...")
    coarse_rain = {}
    coarse_mean = {}
    for i, t in enumerate(sorted(hems), 1):
        rain, _, _ = read_hem(hems[t])
        max_r, mean_r, _ = mapper.aggregate(rain)
        coarse_rain[t] = max_r
        coarse_mean[t] = mean_r
        if i == 1 or i % 10 == 0 or i == len(hems):
            print(f"  {i}/{len(hems)} HEM files processed")

    # Create a real 3-hour future target WITHOUT a pandas merge.
    # CTP grids can contain repeated/invalid coordinates, so merging on
    # timestamp+lat+lon can create a many-to-many explosion in memory.
    # Instead, build each timestamp's target array in the same CTP-cell order
    # and attach it directly to that timestamp's feature frame.
    labelled_frames = []
    label_count = 0

    for t in common:
        future_times = [
            t + pd.Timedelta(minutes=STEP_MINUTES * i)
            for i in range(1, FUTURE_STEPS + 1)
        ]
        available = [ft for ft in future_times if ft in coarse_rain]

        ctp, ctt, emiss, clat, clon = read_ctp(ctps[t])
        current_max = coarse_rain[t]

        frame = pd.DataFrame({
            "timestamp": t,
            "latitude": clat.ravel(),
            "longitude": clon.ravel(),
            "rain_current_max_mm_h": current_max,
            "rain_current_mean_mm_h": coarse_mean[t],
            "cloud_top_pressure_hpa": ctp.ravel(),
            "cloud_top_temperature_k": ctt.ravel(),
            "effective_cloud_emissivity": emiss.ravel(),
        })

        # Only create a label when ALL six future half-hour observations exist.
        if len(available) == FUTURE_STEPS:
            stacked = np.vstack([coarse_rain[ft] for ft in future_times])
            has_any = np.isfinite(stacked).any(axis=0)
            future_max = np.full(stacked.shape[1], np.nan, dtype=np.float32)
            if np.any(has_any):
                safe = np.where(np.isfinite(stacked), stacked, -np.inf)
                future_max[has_any] = np.max(safe[:, has_any], axis=0)

            frame["future_3h_max_mm_h"] = future_max
            frame["event_next_3h"] = np.where(
                np.isfinite(future_max),
                (future_max >= THRESHOLD_MM_H).astype("int8"),
                np.nan,
            )
            label_count += 1
        else:
            frame["future_3h_max_mm_h"] = np.nan
            frame["event_next_3h"] = np.nan

        labelled_frames.append(frame)

    data = pd.concat(labelled_frames, ignore_index=True)
    print(f"Labelled timestamps with complete 3-hour horizon: {label_count}/{len(common)}")

    # Remove clearly invalid CTP values.
    data.loc[
        ~data["cloud_top_pressure_hpa"].between(0, 1100),
        "cloud_top_pressure_hpa"
    ] = np.nan
    data.loc[
        ~data["cloud_top_temperature_k"].between(150, 350),
        "cloud_top_temperature_k"
    ] = np.nan
    data.loc[
        ~data["effective_cloud_emissivity"].between(0, 1.2),
        "effective_cloud_emissivity"
    ] = np.nan

    data.to_csv(out, index=False)

    labelled = data["event_next_3h"].notna()
    print("\nDONE")
    print(f"Rows written: {len(data):,}")
    print(f"Rows with 3-hour labels: {labelled.sum():,}")
    if labelled.any():
        y = data.loc[labelled, "event_next_3h"].astype(int)
        print(f"Positive events: {y.sum():,}")
        print(f"Event rate: {y.mean()*100:.3f}%")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
