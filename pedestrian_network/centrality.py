"""Prepare the attributed model network and sDNA input before QGIS analysis.

Assign segment_id once and preserve it in the shared network.
After calculating sDNA results, run pedestrian_volume.py to attach them and
generate the separate model layers.
"""
import argparse
import re
from pathlib import Path

import geopandas as gpd

from src.utils.config_loader import config_data
from pedestrian_volume import (
    geometry_positions, initialize_network,
    save_network, normalize_identity,
)

CITY = config_data["city_name"]
CITY_DIR = Path(__file__).resolve().parent / "src" / "data" / "output" / CITY


def get_version(path):
    match = re.search(r"_v(\d+)\.(\d+)\.gpkg$", path.name)
    return tuple(map(int, match.groups())) if match else (-1, -1)


def find_latest_street_network():
    # main.py saves POI-enriched networks here; draft is preprocessing only.
    candidates = [
        p for p in CITY_DIR.glob(f"street_net_{CITY}_v*.gpkg")
        if get_version(p) != (-1, -1)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No attributed street network in {CITY_DIR}. Run main.py first; "
            "the draft network does not contain the collected POI data."
        )
    return max(candidates, key=get_version)


def prepare_network(version, source=None):
    path = CITY_DIR / "model" / f"pedestrian_model_network_{CITY}_v{version}.gpkg"
    input_path = CITY_DIR / "draft" / f"centrality_input_{CITY}_v{version}.gpkg"
    if path.exists():
        network = gpd.read_file(path, layer="pedestrian_model_network")
        migrated = normalize_identity(network)
        if set(migrated.columns) != set(network.columns):
            save_network(migrated, path)
            print("Migrated saved IDs to segment_id without renumbering.")
        network = migrated
        if source is not None:
            raise ValueError("This model network already exists. Use a new version for a new source.")
        print(f"Reusing model network and its existing IDs: {path}")
    else:
        source = Path(source) if source else find_latest_street_network()
        streets = gpd.read_file(source)
        required = (
            "laenge [km]",
            "Dienstleistung, Einzelhandel, Gastronomie: Anzahl",
            "Hotels, Pensionen: Anzahl",
        )
        missing = [c for c in required if c not in streets]
        if missing:
            raise ValueError(f"Source lacks model attributes: {missing}. Use the main.py output.")
        if streets.crs is None or streets.geometry.isna().any() or not streets.geometry.is_valid.all():
            raise ValueError("Source has missing/invalid geometries or no CRS.")
        # Adopt the IDs of an existing analyzed input, rather than renumbering it.
        legacy = gpd.read_file(input_path, layer="centrality_input") if input_path.exists() else None
        network = initialize_network(streets, source.name, legacy)
        network["network_version"] = version
        save_network(network, path)
        print(f"Created attributed model network: {path}")
    if input_path.exists():
        legacy = gpd.read_file(input_path, layer="centrality_input")
        migrated = normalize_identity(legacy)
        positions = geometry_positions(network, migrated)
        if not (migrated.iloc[positions].segment_id.to_numpy() == network.segment_id.to_numpy()).all():
            raise ValueError("Existing sDNA input has different segment IDs.")
        if set(migrated.columns) != set(legacy.columns):
            migrated.to_file(input_path, layer="centrality_input", driver="GPKG")
            print("Renamed existing input IDs to segment_id; geometries preserved.")
        print(f"Preserved existing sDNA input: {input_path}")
    else:
        input_path.parent.mkdir(parents=True, exist_ok=True)
        # The input is an export of the canonical streets, not a new network.
        network.to_file(input_path, layer="centrality_input", driver="GPKG")
        print(f"Created sDNA input: {input_path}")
    return network, path, input_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.0", help="Frozen model-network version")
    parser.add_argument("--source", type=Path, help="Specific POI-enriched main.py output")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+", args.version):
        parser.error("--version must be major.minor, e.g. 1.0")
    network, path, input_path = prepare_network(args.version, args.source)
    print(f"City: {CITY}; streets: {len(network):,}")
    print(f"Run sDNA Integral on {input_path} (layer: centrality_input).")
    print("Keep this exact input. Save the actual sDNA result layer as:")
    print(CITY_DIR / "model" / f"centrality_output_{CITY}_v{args.version}.gpkg")
    print("Suggested result layer: centrality_output.")
    print(f"Then run: python pedestrian_volume.py --version {args.version}")
    print("It imports the results, restores segment_id if omitted, and runs the models.")
    print("For another export name, pass --sdna-results and --layer to pedestrian_volume.py.")
    print("Do not regenerate IDs between models. Use a new version if streets change.")


if __name__ == "__main__":
    main()
