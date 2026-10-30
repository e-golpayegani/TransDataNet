from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.config_loader import config_data


# ============================================================
# CONFIGURATION
# ============================================================

CITY = config_data["city_name"]

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "src" / "data" / "output"
CITY_DIR = OUTPUT_ROOT / CITY
MODEL_DIR = CITY_DIR / "model"


# ============================================================
# MODEL
# ============================================================

# Intercept from your already-fitted Negative Binomial model
INTERCEPT_CL = 6.263
INTERCEPT_PoI = 6.923


# Add/remove variables here depending on the model you want to apply.
#
# coefficient = coefficient from your fitted NB model
# transform:
#   "none"       -> use the variable directly
#   "per_length" -> divide by (length_km * 10)

MODEL_VARIABLES = {
'''
    # Example: closeness/accessibility only
    "NQPDA500": {
        "source_col": "NQPDA500",
        "coefficient": 0.904,   # <-- PUT YOUR COEFFICIENT HERE
        "transform": "none",
    },
'''
    # Uncomment if included in your model:
    #
    "service_retail_gastronomy": {
        "source_col":
            "Dienstleistung, Einzelhandel, Gastronomie: Anzahl",
        "coefficient": 0.008,
        "transform": "per_length",
    },
    
    "hotels": {
        "source_col":
            "Hotels, Pensionen: Anzahl",
        "coefficient": 0.095,
        "transform": "per_length",
    },
}


LENGTH_COLUMN = "laenge [km]"
PER_LENGTH_FACTOR = 10.0


# ============================================================
# FIND LATEST INPUT NETWORK
# ============================================================

def get_version(path):

    match = re.search(
        r"_v(\d+)\.(\d+)\.gpkg$",
        path.name
    )

    if match is None:
        return -1, -1

    return int(match.group(1)), int(match.group(2))


def find_latest_centrality_network():

    files = list(
        MODEL_DIR.glob(
            f"Centrality_output_{CITY}.shp"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No centrality network found in:\n{MODEL_DIR}"
        )

    return max(files, key=get_version)


# ============================================================
# PREPARE VARIABLE
# ============================================================

def prepare_variable(gdf, variable_name, settings):

    source_col = settings["source_col"]
    transform = settings.get("transform", "none")

    if source_col not in gdf.columns:
        raise KeyError(
            f"Column '{source_col}' required for "
            f"'{variable_name}' was not found."
        )

    values = pd.to_numeric(
        gdf[source_col],
        errors="coerce"
    )

    if transform == "none":

        return values

    elif transform == "per_length":

        if LENGTH_COLUMN not in gdf.columns:
            raise KeyError(
                f"Length column '{LENGTH_COLUMN}' was not found."
            )

        length = pd.to_numeric(
            gdf[LENGTH_COLUMN],
            errors="coerce"
        )

        denominator = (
            length * PER_LENGTH_FACTOR
        ).replace(0, np.nan)

        return values / denominator

    else:

        raise ValueError(
            f"Unknown transformation: {transform}"
        )


# ============================================================
# APPLY EXISTING NB MODEL
# ============================================================

def calculate_pedestrian_volume(gdf):

    # Start with the intercept
    linear_predictor = pd.Series(
        # INTERCEPT_CL,
        INTERCEPT_PoI,
        index=gdf.index,
        dtype=float
    )

    print("\nModel:")
    #print(f"Intercept = {INTERCEPT_CL}")
    print(f"Intercept = {INTERCEPT_PoI}")
    # Add beta * X for every selected variable
    for variable_name, settings in MODEL_VARIABLES.items():

        coefficient = settings["coefficient"]

        x = prepare_variable(
            gdf,
            variable_name,
            settings
        )

        # Optional: save the exact transformed variable
        # that was fed into the model
        model_col = f"model_{variable_name}"
        gdf[model_col] = x

        linear_predictor += coefficient * x

        print(
            f"{variable_name}: "
            f"beta = {coefficient}"
        )

    # --------------------------------------------------------
    # NB expected pedestrian volume
    # --------------------------------------------------------

    gdf["PV_linear_predictor"] = linear_predictor

    gdf["PV"] = np.exp(
        linear_predictor
    )

    # Invalid model inputs remain NaN
    gdf["PV"] = gdf["PV"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Rounded predicted pedestrian count
    gdf["PV"] = (
        gdf["PV"]
        .round()
        .astype("Int64")
    )

    return gdf


# ============================================================
# SAVE
# ============================================================

def save_output(gdf):

    output_path = (
        MODEL_DIR /
        f"Pedestrian_volume_PoI_{CITY}_v1.0.gpkg"
    )

    gdf.to_file(
        output_path,
        layer="pedestrian_volume_PoI",
        driver="GPKG"
    )

    print("\nOutput saved:")
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n====================================")
    print("PEDESTRIAN VOLUME")
    print("====================================")

    input_path = find_latest_centrality_network()

    print("\nInput:")
    print(input_path)

    gdf = gpd.read_file(input_path)

    print(f"Street segments: {len(gdf)}")

    gdf = calculate_pedestrian_volume(gdf)

    save_output(gdf)

    print("\nDone.")


if __name__ == "__main__":
    main()