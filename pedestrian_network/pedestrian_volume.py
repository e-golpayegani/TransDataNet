"""Apply separate fitted models to the SAME attributed street network.

After QGIS analysis, this script imports sDNA results and restores omitted IDs.
Inputs remain on pedestrian_model_network. Final predictions are exported
as separate model layers plus a comparison layer, all keyed by segment_id.
Coefficients below are carried over from the existing script; use only with the
same predictor definitions, units and transformations as the fitted models.
Add further fitted models as separate MODELS entries, each with its own intercept.
Every selected entry creates its own layer with whole-number pedestrian volumes.
"""
import argparse
from itertools import combinations
import json
import re
import os
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src.utils.config_loader import config_data

CITY = config_data["city_name"]
MODEL_DIR = Path(__file__).resolve().parent / "src" / "data" / "output" / CITY / "model"
LENGTH_COLUMN = "laenge [km]"
PER_LENGTH_FACTOR = 10.0

# Each unique model name creates model_<name> and PV_<name> automatically.
# Add any number of fitted POI models here, each with its own intercept/variables.
MODELS = {
    "closeness": {
        "requires_sdna": True,
        "intercept": 6.263,
        "variables": {
            "NQPDA500": {
                "source_col": "NQPDA500",
                "coefficient": 0.904,
                "transform": "none",
            },
        },
    },
    "poi": {
        "intercept": 6.923,
        "variables": {
            "service_retail_gastronomy": {
                "source_col": "Dienstleistung, Einzelhandel, Gastronomie: Anzahl",
                "coefficient": 0.008,
                "transform": "per_length",
            },
            "hotels": {
                "source_col": "Hotels, Pensionen: Anzahl",
                "coefficient": 0.095,
                "transform": "per_length",
            },
        },
    },
}


# Segment identity and sDNA matching are kept here for both main scripts.
def geometry_positions(base, incoming):
    """Return incoming row positions in base order using exact normalized XY."""
    if base.crs is None or incoming.crs is None or base.crs != incoming.crs:
        raise ValueError("Both networks must have the same declared CRS.")
    if base.empty or len(base) != len(incoming):
        raise ValueError("Network feature counts differ or the network is empty.")

    def keys(frame):
        if frame.geometry.isna().any() or frame.geometry.is_empty.any():
            raise ValueError("Missing/empty geometry prevents reliable matching.")
        result = pd.Index(shapely.to_wkb(
            shapely.normalize(shapely.force_2d(frame.geometry.array))
        ))
        if not result.is_unique:
            raise ValueError("Duplicate geometries make the match ambiguous.")
        return result

    positions = keys(incoming).get_indexer(keys(base))
    if (positions < 0).any():
        raise ValueError("Street geometries differ; use the exact network analyzed.")
    return positions


def validate_identity(frame):
    if (
        "segment_id" not in frame
        or frame.segment_id.isna().any()
        or not frame.segment_id.is_unique
        or frame.segment_id.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("segment_id must be present, unique and nonmissing.")
    if {"street_id", "centrality_id"} & set(frame.columns):
        raise ValueError("Use normalize_identity to migrate legacy ID fields first.")


def normalize_identity(frame):
    """Migrate old names without generating or renumbering IDs.

    Prefer existing segment_id, then the old numeric centrality_id, then street_id.
    Remove the redundant old fields. A conflicting numeric alias is an error.
    IDs are unique within a city/network version, not across unrelated networks.
    """
    output = frame.copy()
    if "segment_id" in output:
        if "centrality_id" in output and not np.array_equal(
            output.segment_id.to_numpy(), output.centrality_id.to_numpy()
        ):
            raise ValueError("segment_id conflicts with legacy centrality_id.")
    else:
        legacy = next((c for c in ("centrality_id", "street_id") if c in output), None)
        if legacy is None:
            raise ValueError("No saved segment identity; run centrality.py to initialize it.")
        output = output.rename(columns={legacy: "segment_id"})
    output = output.drop(columns=["street_id", "centrality_id"], errors="ignore")
    validate_identity(output)
    return output


def initialize_network(streets, source_name, legacy_input=None):
    """Assign segment IDs once; adopt existing analyzed-input IDs if available."""
    if streets.empty:
        raise ValueError("Cannot initialize an empty network.")
    frame = streets.copy()
    if legacy_input is not None:
        legacy = normalize_identity(legacy_input)
        positions = geometry_positions(frame, legacy)
        existing = normalize_identity(frame) if (
            {"segment_id", "centrality_id", "street_id"} & set(frame.columns)
        ) else None
        ids = legacy.iloc[positions].segment_id.to_numpy()
        if existing is not None and not np.array_equal(existing.segment_id.to_numpy(), ids):
            raise ValueError("Source IDs conflict with the saved sDNA input.")
        frame = frame.drop(columns=["street_id", "centrality_id"], errors="ignore")
        frame["segment_id"] = ids
    elif {"segment_id", "centrality_id", "street_id"} & set(frame.columns):
        frame = normalize_identity(frame)
    else:
        frame["segment_id"] = np.arange(1, len(frame) + 1)
    validate_identity(frame)
    frame["network_source"] = source_name
    return frame


def attach_sdna_results(network, results):
    """Attach calculations to reference segments, restoring IDs sDNA omitted.

    Existing segment_id (or legacy centrality_id) is validated against geometry.
    If neither is present, require a complete, unique exact XY geometry match.
    Never infer identity from row order or sDNA's generated ID.
    """
    network = normalize_identity(network)
    result_id = next((c for c in ("segment_id", "centrality_id") if c in results), None)
    if result_id:
        ids = results[result_id]
        if ids.isna().any() or not ids.is_unique:
            raise ValueError("Result segment IDs are missing or duplicated.")
        if len(network) != len(results):
            raise ValueError("Result feature count differs from the model network.")
        positions = pd.Index(ids).get_indexer(network.segment_id)
        if (positions < 0).any():
            raise ValueError("Result segment IDs do not match the model network.")
        ordered = results.iloc[positions]
        if network.crs is None or network.crs != results.crs:
            raise ValueError("Result CRS differs from the model network.")
        if ordered.geometry.isna().any() or ordered.geometry.is_empty.any():
            raise ValueError("Missing/empty result geometry prevents verification.")
        if not shapely.equals(
            shapely.force_2d(network.geometry.array),
            shapely.force_2d(ordered.geometry.array),
        ).all():
            raise ValueError("Result IDs point to different geometries.")
        if {"segment_id", "centrality_id"} <= set(results.columns) and not np.array_equal(
            ordered.segment_id.to_numpy(), ordered.centrality_id.to_numpy()
        ):
            raise ValueError("Result segment_id conflicts with legacy centrality_id.")
    else:
        positions = geometry_positions(network, results)
        ordered = results.iloc[positions]

    output = network.copy()
    stale = [c for c in output if c.startswith(("PV_", "model_", "comparison_"))]
    output = output.drop(columns=stale)
    protected = {
        network.geometry.name, results.geometry.name, "segment_id", "street_id",
        "centrality_id", "centrality_source", "network_source", "network_version",
        "sdna_source", "ID",
    }
    for column in results:
        if column in protected:
            continue
        values = ordered[column].to_numpy()
        if column in network:
            if column in network.attrs.get("sdna_columns", []):
                output[column] = values
            elif not pd.Series(values).equals(network[column].reset_index(drop=True)):
                raise ValueError(f"Result would overwrite original attribute: {column}")
        else:
            output[column] = values
    # Remove a previously imported sDNA row ID, but preserve unrelated source data.
    if "ID" in network.attrs.get("sdna_columns", []):
        output = output.drop(columns=["ID"], errors="ignore")
    validate_identity(output)
    return output


def save_network(network, path):
    validate_identity(network)
    path.parent.mkdir(parents=True, exist_ok=True)
    network.to_file(path, layer="pedestrian_model_network", driver="GPKG")


def prepare_variable(gdf, variable_name, settings):
    column = settings["source_col"]
    if column not in gdf:
        raise KeyError(f"Missing predictor '{column}' for '{variable_name}'.")
    values = pd.to_numeric(gdf[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    transform = settings.get("transform", "none")
    if transform == "none":
        return values
    if transform == "per_length":
        if LENGTH_COLUMN not in gdf:
            raise KeyError(f"Missing length column: {LENGTH_COLUMN}")
        length = pd.to_numeric(gdf[LENGTH_COLUMN], errors="coerce")
        denominator = (length * PER_LENGTH_FACTOR).where(np.isfinite(length) & (length > 0))
        return values / denominator
    raise ValueError(f"Unknown transformation: {transform}")


def calculate_pedestrian_volume(gdf, selected=None):
    gdf = normalize_identity(gdf)
    selected = list(MODELS) if selected is None else list(dict.fromkeys(selected))
    if not selected or any(name not in MODELS for name in selected):
        raise ValueError("Select at least one defined model.")
    output = gdf.copy()
    # Each output contains only predictions calculated during this run.
    output = output.drop(columns=[
        c for c in output if c == "PV" or c.startswith(("PV_", "model_", "comparison_"))
    ])
    for name in selected:
        model = MODELS[name]
        eta = pd.Series(model["intercept"], index=output.index, dtype=float)
        for variable, settings in model["variables"].items():
            values = prepare_variable(output, variable, settings)
            output[f"model_{name}_{variable}"] = values
            eta += settings["coefficient"] * values
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = np.exp(eta)
        predicted = predicted.where(np.isfinite(predicted))
        rounded = predicted.round()
        rounded = rounded.where((rounded >= 0) & (rounded < 2**63))
        output[f"PV_{name}"] = rounded.astype("Int64")
        output[f"PV_{name}_valid"] = output[f"PV_{name}"].notna()
        print(f"{name}: {output[f'PV_{name}'].notna().sum():,}/{len(output):,} valid predictions")
    if len(selected) > 1:
        output["comparison_valid"] = output[
            [f"PV_{name}" for name in selected]
        ].notna().all(axis=1)
        for first, second in combinations(selected, 2):
            output[f"comparison_{first}_minus_{second}"] = (
                output[f"PV_{first}"] - output[f"PV_{second}"]
            )
    return output



def save_model_layers(output, selected, output_path):
    """Write one layer per selected model and a layer for paired comparisons.

    Build a complete new GeoPackage before replacing a previous export, so
    partial runs cannot leave stale model layers in the results file.
    The canonical input network is never overwritten here.
    """
    output = normalize_identity(output)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".model_export_{uuid.uuid4().hex}.gpkg")
    try:
        for name in dict.fromkeys(selected):
            own_fields = {
                f"PV_{name}", f"PV_{name}_valid",
                *(f"model_{name}_{variable}" for variable in MODELS[name]["variables"]),
            }
            columns = [
                column for column in output
                if not column.startswith(("PV_", "model_", "comparison_"))
                or column in own_fields
            ]
            layer = output[columns].copy()
            layer.to_file(staging, layer=f"model_{name}", driver="GPKG")
        output.to_file(staging, layer="model_comparison", driver="GPKG")
        os.replace(staging, output_path)
    finally:
        if staging.exists():
            staging.unlink()
    return output_path


def load_model_network(path, sdna_path=None, sdna_layer="centrality_output"):
    """Import and validate manual sDNA results before fitting any predictions."""
    network = gpd.read_file(path, layer="pedestrian_model_network")
    migrated = normalize_identity(network)
    if sdna_path is not None:
        sdna_path = Path(sdna_path)
        if sdna_path.resolve() == Path(path).resolve():
            raise ValueError("Supply the separate sDNA export, not the shared model network.")
        if not sdna_path.exists():
            raise FileNotFoundError(f"sDNA result file not found: {sdna_path}")
        results = gpd.read_file(sdna_path, layer=sdna_layer)
        metadata_path = Path(path).with_suffix(".sdna.json")
        previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else []
        migrated.attrs["sdna_columns"] = previous
        merged = attach_sdna_results(migrated, results)
        columns = sorted(set(previous) | (set(merged.columns) - set(migrated.columns)))
        merged["sdna_source"] = str(sdna_path.resolve())
        save_network(merged, path)
        metadata_path.write_text(json.dumps(columns, indent=2), encoding="utf-8")
        print(f"Attached sDNA values and preserved segment_id for {len(merged):,} segments.")
        return merged
    if set(migrated.columns) != set(network.columns):
        save_network(migrated, path)
    return migrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--sdna-results", type=Path, help="Optional custom path to the manual sDNA export")
    parser.add_argument("--layer", default="centrality_output", help="sDNA export layer name")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+", args.version):
        parser.error("--version must be major.minor, e.g. 1.0")
    path = MODEL_DIR / f"pedestrian_model_network_{CITY}_v{args.version}.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run centrality.py first.")
    needs_sdna = any(MODELS[name].get("requires_sdna", False) for name in args.models)
    sdna_path = args.sdna_results
    if sdna_path is None and needs_sdna:
        suggested = MODEL_DIR / f"centrality_output_{CITY}_v{args.version}.gpkg"
        if suggested.exists():
            sdna_path = suggested
    network = load_model_network(path, sdna_path, args.layer)
    if needs_sdna and sdna_path is None:
        required = {
            settings["source_col"]
            for name in args.models if MODELS[name].get("requires_sdna", False)
            for settings in MODELS[name]["variables"].values()
        }
        missing = sorted(required - set(network.columns))
        if missing:
            raise FileNotFoundError(
                f"Closeness inputs are missing: {missing}. Export the sDNA result layer "
                f"to {MODEL_DIR / f'centrality_output_{CITY}_v{args.version}.gpkg'} "
                "(layer centrality_output), or pass --sdna-results and --layer."
            )
    output = calculate_pedestrian_volume(network, args.models)
    result_path = MODEL_DIR / f"pedestrian_volume_{CITY}_v{args.version}.gpkg"
    save_model_layers(output, args.models, result_path)
    print(f"Saved model layers and model_comparison: {result_path}")
    print(f"Shared attributed input: {path}")


if __name__ == "__main__":
    main()
