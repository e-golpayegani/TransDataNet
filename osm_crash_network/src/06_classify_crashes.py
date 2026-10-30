from pathlib import Path
import re
import ast
import shutil
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd

# ============================================================
# 04_classify_and_aggregate_crashes.py
#
# Purpose:
# 1. Load crash shapefiles for multiple years
# 2. Keep only crashes for the city of interest
# 3. Classify crashes as intersection or segment crashes
# 4. Assign intersection crashes to junction_id
# 5. Assign segment crashes to nearest segment_id
# 6. Aggregate crash counts by type, mode, severity, year, etc.
# ============================================================

# -----------------------------
# User settings
# -----------------------------
CITY = "Dresden"

DATA_ROOT = Path(
    r"Z:\_Public\Promotionen\Golpayegani\Paper3\03_OSM_Network_Simplification\GIS_Data"
)

CRASH_DIR = Path(r"Z:\_Public\Promotionen\Golpayegani\Paper3\02_Database\Dependent_variable\Dresden\Unfallatlas\Shape\Unfall_2022-2025")

CRASH_2022_SHP = CRASH_DIR.parent / "Unfallorte2022_EPSG25832_Shape" / "shp" / "Unfallorte2022_LinRef.shp"

CRASH_CSV_DIR = CRASH_DIR.parent.parent / "CSV" / CRASH_DIR.name

# Dresden municipality code in the Unfallatlas dataset.
# If this does not match your data, print UGEMEINDE value_counts first.
DRESDEN_UGEMEINDE = 146120000
DRESDEN_ULAND = 14
DRESDEN_UREGBEZ = 6
DRESDEN_UKREIS = 12

MAX_SEGMENT_SNAP_DISTANCE_M = 30
YEARS = [2022, 2023, 2024, 2025]
MODE_COLUMNS = {"IstPKW": "car", "IstRad": "bicycle", "IstFuss": "pedestrian",
                "IstKrad": "motorcycle", "IstGkfz": "goods_vehicle", "IstSonstig": "other"}

# Fields to aggregate from the German crash database
AGGREGATE_COLUMNS = [
    "UKATEGORIE",
    "UART",
    "UTYP1",
    "ULICHTVERH",
    "IstRad",
    "IstPKW",
    "IstFuss",
    "IstKrad",
    "IstSonstig",
    "IstGkfz",
    "year",
    "UMONAT",
    "USTUNDE",
    "UWOCHENTAG",
    "mode_type",
    "severity_type",
    "light_type",
    "weekend",
]

# -----------------------------
# Input / output paths
# -----------------------------
CITY_DIR = DATA_ROOT / CITY
PROCESSED_DIR = CITY_DIR / "processed"
OUTPUT_DIR = CITY_DIR / "outputs" / "crash_analysis_rebuilt_20261030"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JUNCTION_GPKG = PROCESSED_DIR / f"{CITY.lower()}_junction_areas.gpkg"
SEGMENT_GPKG = PROCESSED_DIR / f"{CITY.lower()}_unified_street_network_rebuilt_20261030.gpkg"
ORIGINAL_GPKG = PROCESSED_DIR / f"{CITY.lower()}_segment_network.gpkg"
SOURCE_GPKG = PROCESSED_DIR / f"{CITY.lower()}_j2j_detailed_network_rebuilt_20261030.gpkg"

OUTPUT_GPKG = OUTPUT_DIR / f"{CITY.lower()}_classified_and_aggregated_crashes.gpkg"

# -----------------------------
# Helper functions
# -----------------------------
def extract_year(path: Path):
    match = re.search(r"(20\d{2})", path.name)
    return int(match.group(1)) if match else None


def yes(value):
    return int(value) == 1 if pd.notna(value) else False


def derive_mode_type(row):
    car = yes(row.get("IstPKW", 0))
    bike = yes(row.get("IstRad", 0))
    ped = yes(row.get("IstFuss", 0))
    moto = yes(row.get("IstKrad", 0))
    heavy = yes(row.get("IstGkfz", 0))
    other = yes(row.get("IstSonstig", 0))
    '''
    if car and bike:
        return "car_bike"
    if car and ped:
        return "car_pedestrian"
    if car and moto:
        return "car_motorcycle"
    if car and heavy:
        return "car_heavy_vehicle"
    if bike and ped:
        return "bike_pedestrian"
    if bike and moto:
        return "bike_motorcycle"
    if heavy and ped:
        return "heavy_vehicle_pedestrian"
    if heavy and bike:
        return "heavy_vehicle_bike"
    '''
    if ped:
        return "pedestrian_involved"
    if bike:
        return "bike_involved"
    if moto:
        return "motorcycle_involved"
    if heavy:
        return "heavy_vehicle_involved"
    if car:
        return "car_involved"
    if other:
        return "other_involved"

    return "unknown"


def derive_severity_type(value):
    # German Unfallatlas commonly uses:
    # 1 = fatal
    # 2 = serious injury
    # 3 = slight injury
    if pd.isna(value):
        return "unknown"

    value = int(value)

    if value == 1:
        return "fatal"
    if value == 2:
        return "serious_injury"
    if value == 3:
        return "slight_injury"

    return f"severity_{value}"


def derive_light_type(value):
    # Common coding:
    # 0 = daylight
    # 1 = twilight
    # 2 = darkness
    if pd.isna(value):
        return "unknown"

    value = int(value)

    if value == 0:
        return "daylight"
    if value == 1:
        return "twilight"
    if value == 2:
        return "darkness"

    return f"light_{value}"


def clean_name(value):
    value = str(value)
    value = value.strip()
    value = value.replace(" ", "_")
    value = value.replace("-", "_")
    value = value.replace("/", "_")
    value = value.replace(".", "_")
    value = value.replace(":", "_")
    value = value.replace(";", "_")
    value = value.replace(",", "_")
    value = value.replace("(", "")
    value = value.replace(")", "")
    return value


def aggregate_counts(gdf, group_id_col, prefix):
    """
    Creates one wide table with:
    - total crashes
    - counts by selected crash attributes
    """

    outputs = []

    total = (
        gdf.groupby(group_id_col)
        .size()
        .reset_index(name=f"{prefix}_total_crashes")
    )
    outputs.append(total)

    for col in AGGREGATE_COLUMNS:
        if col not in gdf.columns:
            continue

        temp = (
            gdf.groupby([group_id_col, col])
            .size()
            .reset_index(name="count")
        )

        if temp.empty:
            continue

        wide = temp.pivot(
            index=group_id_col,
            columns=col,
            values="count"
        ).fillna(0)

        wide.columns = [
            f"{prefix}_{col}_{clean_name(c)}"
            for c in wide.columns
        ]

        wide = wide.reset_index()
        outputs.append(wide)

    merged = outputs[0]

    for table in outputs[1:]:
        merged = merged.merge(table, on=group_id_col, how="left")

    return merged.fillna(0)


def normalize_crash_columns(frame):
    frame.columns = frame.columns.str.strip().str.lstrip("\ufeff")
    if "IstSonstige" in frame and "IstSonstig" not in frame:
        frame = frame.rename(columns={"IstSonstige": "IstSonstig"})
    return frame


def load_csv_reference(directory):
    files = sorted(path for path in directory.rglob("*") if path.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No Unfallatlas CSV files found in {directory}")
    frames, inventory = [], []
    required = ["ULAND", "UREGBEZ", "UKREIS", "UJAHR", "UKATEGORIE", "ULICHTVERH"]
    for file in files:
        for encoding in ["utf-8-sig", "cp1252"]:
            try:
                selected, loaded = [], 0
                for chunk in pd.read_csv(file, sep=";", encoding=encoding, dtype=str, chunksize=100000):
                    chunk = normalize_crash_columns(chunk)
                    missing = [col for col in required if col not in chunk]
                    if missing:
                        raise ValueError(f"CSV {file.name}: missing columns {missing}")
                    loaded += len(chunk)
                    chunk["year"] = pd.to_numeric(chunk["UJAHR"], errors="coerce")
                    mask = (pd.to_numeric(chunk["ULAND"], errors="coerce").eq(DRESDEN_ULAND)
                            & pd.to_numeric(chunk["UREGBEZ"], errors="coerce").eq(DRESDEN_UREGBEZ)
                            & pd.to_numeric(chunk["UKREIS"], errors="coerce").eq(DRESDEN_UKREIS)
                            & chunk["year"].isin(YEARS))
                    selected.append(chunk.loc[mask].assign(source_csv=file.name))
                break
            except UnicodeDecodeError:
                if encoding == "cp1252":
                    raise
        frame = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
        frames.append(frame)
        inventory.append({"csv_file": file.name, "loaded_rows": loaded, "dresden_2022_2025_rows": len(frame)})
    reference = pd.concat(frames, ignore_index=True)
    if reference.empty:
        raise ValueError("No Dresden 2022-2025 crashes found in CSV reference files.")
    reference["year"] = reference["year"].astype(int)
    if "UIDENTSTLAE" in reference:
        duplicates = reference.duplicated(["year", "UIDENTSTLAE"], keep=False) & reference.UIDENTSTLAE.notna()
        if duplicates.any():
            raise ValueError(f"CSV reference contains {int(duplicates.sum())} rows with repeated crash IDs; check overlapping files.")
    return reference, pd.DataFrame(inventory)


def join_ids(values):
    identifiers = set()
    for value in values:
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            if pd.notna(item):
                identifiers.update(str(item).split("; "))
    return "; ".join(sorted(identifiers))


def id_values(value):
    """Read the semicolon-separated lineage fields without losing integer precision."""
    if isinstance(value, (list, tuple, set)):
        return {item for part in value for item in id_values(part)}
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if text.startswith("["):
        return id_values(ast.literal_eval(text))
    return {part.strip() for part in re.split(r"[;,]", text) if part.strip()}


def serialize_ids(values):
    return "; ".join(sorted(set(values)))


def build_network_inventory(segments, junctions, source_edges, original_edges):
    """Trace explicit source mappings; do not infer ancestry from spatial overlap."""
    import osmnx as ox

    raw_path = CITY_DIR / "raw" / f"{CITY.lower()}_raw_osm_network.graphml"
    raw = ox.load_graphml(raw_path)
    for u, v, key, data in raw.edges(keys=True, data=True):
        data["_level0_edge_id"] = f"{u}_{v}_{key}"
    simplified = ox.simplify_graph(raw)
    raw_lineage = {
        f"{u}_{v}_{key}": id_values(data["_level0_edge_id"])
        for u, v, key, data in simplified.edges(keys=True, data=True)
    }
    # Detect mismatched snapshots or simplification settings rather than invent lineage.
    for edge in original_edges.itertuples():
        graph_id = edge.graph_edge_ids
        if graph_id not in raw_lineage:
            raise ValueError(f"Level 1 edge {graph_id} not found in reconstructed raw graph.")
        data = simplified[int(edge.u)][int(edge.v)][int(edge.key)]
        if id_values(data["osmid"]) != id_values(edge.osmid):
            raise ValueError(f"OSM snapshot mismatch for Level 1 edge {graph_id}.")
        raw_geometry = data.get("geometry")
        if raw_geometry is not None and not raw_geometry.equals_exact(edge.geometry, 0.001):
            raise ValueError(f"Level 1 geometry differs from raw simplification: {graph_id}")

    source_lookup = source_edges.set_index("source_edge_id")
    candidates = gpd.read_file(JUNCTION_GPKG, layer="candidate_junction_nodes").to_crs(junctions.crs)
    links = gpd.sjoin(
        candidates[["candidate_node_id", "osmid", "geometry"]],
        junctions[["junction_id", "geometry"]], how="inner", predicate="intersects",
    )
    if links.candidate_node_id.duplicated().any() or len(links) != len(candidates):
        raise ValueError("Every candidate junction node must belong to exactly one junction.")
    candidate_groups = {key: group for key, group in links.groupby("junction_id")}

    rows = []
    for feature in segments.itertuples():
        level2_ids = id_values(feature.source_edge_ids)
        source = source_lookup.loc[[int(value) for value in level2_ids]]
        level1_ids = set(source.graph_edge_ids)
        level0_ids = set().union(*(raw_lineage[value] for value in level1_ids))
        level3_ids = id_values(feature.source_j2j_ids)
        if id_values(feature.osmid) != set().union(*(id_values(value) for value in source.osmid)):
            raise ValueError(f"Level 4 OSM way provenance mismatch: {feature.level4_id}")
        rows.append({
            "location_type": "segment", "network_id": feature.level4_id,
            "level4_id": feature.level4_id, "level4_length_m": feature.geometry.length,
            "level3_j2j_ids": serialize_ids(level3_ids), "n_level3_edges": len(level3_ids),
            # Level 2 source_edge_id is the index assigned in script 03's debug layer.
            "level2_relevant_edge_ids": serialize_ids(level2_ids), "n_level2_edges": len(level2_ids),
            "level1_edge_ids": serialize_ids(level1_ids), "n_level1_edges": len(level1_ids),
            "level0_raw_edge_ids": serialize_ids(level0_ids), "n_level0_edges": len(level0_ids),
            "original_osm_way_ids": join_ids(source.original_osm_way_ids),
            "network_geometry_wkt": feature.geometry.wkt, "network_geometry_crs": str(segments.crs),
        })
    for feature in junctions.itertuples():
        group = candidate_groups.get(feature.junction_id)
        node_ids = set() if group is None else {str(int(value)) for value in group.candidate_node_id}
        osm_node_ids = set() if group is None else {str(int(value)) for value in group.osmid}
        rows.append({
            "location_type": "intersection", "network_id": feature.junction_id,
            "junction_id": feature.junction_id, "junction_area_m2": feature.geometry.area,
            "level2_candidate_node_ids": serialize_ids(node_ids),
            "n_candidate_junction_nodes": len(node_ids),
            "candidate_original_osm_node_ids": serialize_ids(osm_node_ids),
            "original_osm_way_ids": feature.original_osm_way_ids,
            "network_geometry_wkt": feature.geometry.wkt, "network_geometry_crs": str(junctions.crs),
        })
    inventory = pd.DataFrame(rows)
    integer_columns = ["network_id", "level4_id", "junction_id", "n_level3_edges",
                       "n_level2_edges", "n_level1_edges", "n_level0_edges",
                       "n_candidate_junction_nodes"]
    inventory[integer_columns] = inventory[integer_columns].astype("Int64")
    if inventory.duplicated(["location_type", "network_id"]).any():
        raise ValueError("Duplicate network feature in inventory.")
    return inventory


def export_counts(frame, inventory):
    """One row per network feature, including features without crashes."""
    result = inventory.copy().set_index(["location_type", "network_id"])
    assigned = frame.loc[frame.assignment.ne("none")].copy()
    assigned["location_type"] = assigned.assignment
    assigned["network_id"] = assigned.segment_id.where(
        assigned.assignment.eq("segment"), assigned.junction_id
    )
    group_keys = ["location_type", "network_id"]

    def add_count(name, mask):
        counts = assigned.loc[mask].groupby(group_keys).size()
        result[name] = counts.reindex(result.index, fill_value=0).astype(int)

    add_count("total_crashes_2022_2025", assigned.year.isin(YEARS))
    for year in YEARS:
        add_count(f"crashes_{year}", assigned.year.eq(year))
    fields = {"UKATEGORIE": range(1, 4), "UART": range(10), "UTYP1": range(1, 8),
              **{col: [1] for col in MODE_COLUMNS}}
    for field, codes in fields.items():
        values = pd.to_numeric(assigned[field], errors="coerce") if field in assigned else None
        for code in codes:
            name = f"crashes_with_{MODE_COLUMNS[field]}" if field in MODE_COLUMNS else f"crashes_{field}_{code}"
            if values is None:
                result[name] = pd.NA
            else:
                add_count(name, values.eq(code))
    if "IstStrassenzustand" in assigned:
        surface = assigned.groupby(group_keys)["IstStrassenzustand"].agg(
            lambda values: serialize_ids({str(int(value)) for value in
                                         pd.to_numeric(values, errors="coerce").dropna()})
        )
        result["road_surface_condition_codes"] = surface.reindex(result.index, fill_value="")
    else:
        # Use the full CSV field name; shapefile names may be truncated.
        result["road_surface_condition_codes"] = pd.NA
    result = result.reset_index()
    assert result.total_crashes_2022_2025.sum() == len(assigned)
    assert result[[f"crashes_{year}" for year in YEARS]].sum(axis=1).equals(result.total_crashes_2022_2025)
    return result


# ============================================================
# Load crash data
# ============================================================
print("=" * 60)
print("Loading crash shapefiles")
print("=" * 60)

# The combined folder contains bare .shp copies; use complete yearly datasets.
crash_files = []
for year in YEARS:
    directory = CRASH_DIR.parent / f"Unfallorte{year}_EPSG25832_Shape" / "shp"
    files = sorted(directory.glob("*.shp"))
    if len(files) != 1:
        raise ValueError(f"Expected one complete {year} shapefile in {directory}, found {len(files)}")
    for extension in [".shx", ".dbf", ".prj"]:
        if not files[0].with_suffix(extension).exists():
            raise FileNotFoundError(f"Missing shapefile companion: {files[0].with_suffix(extension)}")
    crash_files.extend(files)

if not crash_files:
    raise FileNotFoundError(f"No crash shapefiles found in:\n{CRASH_DIR}")

crash_layers = []

for file in crash_files:
    print(f"Reading: {file}")
    # Stage one year locally: scanning a large DBF over a mapped drive is slow.
    with TemporaryDirectory(prefix="dresden_crash_input_") as temporary:
        local_file = Path(temporary) / file.name
        for extension in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            companion = file.with_suffix(extension)
            if companion.exists():
                shutil.copyfile(companion, local_file.with_suffix(extension))
        print("  Local input ready; selecting Dresden records...")
        gdf = normalize_crash_columns(gpd.read_file(local_file, where="CAST(ULAND AS INTEGER) = 14 AND CAST(UREGBEZ AS INTEGER) = 6 AND CAST(UKREIS AS INTEGER) = 12"))
    print(f"  Dresden records read: {len(gdf):,}")
    gdf["source_file"] = file.name

    file_year = extract_year(file)
    if "UJAHR" in gdf.columns:
        gdf["year"] = gdf["UJAHR"]
    else:
        gdf["year"] = file_year

    if gdf.crs is None:
        raise ValueError(f"Crash shapefile has no CRS: {file}")
    if crash_layers and gdf.crs != crash_layers[0].crs:
        gdf = gdf.to_crs(crash_layers[0].crs)
    crash_layers.append(gdf)

crashes = pd.concat(crash_layers, ignore_index=True)
crashes = gpd.GeoDataFrame(
    crashes,
    geometry="geometry",
    crs=crash_layers[0].crs,
)

crashes = crashes.reset_index(drop=True)
crashes["crash_id"] = crashes.index + 1

print(f"Total crashes loaded: {len(crashes):,}")
print(f"Original CRS: {crashes.crs}")

# -----------------------------
# Keep only Dresden crashes
# -----------------------------
city_columns = ["ULAND", "UREGBEZ", "UKREIS"]
missing_city_columns = [col for col in city_columns if col not in crashes.columns]

if not missing_city_columns:
    print("Filtering crashes by ULAND, UREGBEZ, and UKREIS...")
    print("Top administrative code combinations:")
    print(crashes[city_columns].value_counts().head(10))

    before = len(crashes)

    crashes = crashes[
        (pd.to_numeric(crashes["ULAND"], errors="coerce") == DRESDEN_ULAND)
        & (pd.to_numeric(crashes["UREGBEZ"], errors="coerce") == DRESDEN_UREGBEZ)
        & (pd.to_numeric(crashes["UKREIS"], errors="coerce") == DRESDEN_UKREIS)
    ].copy()

    after = len(crashes)

    print(f"Crashes before city filter: {before:,}")
    print(f"Crashes after city filter: {after:,}")

    if after == 0:
        raise ValueError(
            "No crashes remain after filtering by ULAND, UREGBEZ, and UKREIS. "
            "Check DRESDEN_ULAND, DRESDEN_UREGBEZ, and DRESDEN_UKREIS."
        )
else:
    raise ValueError(f"Cannot filter Dresden: missing columns {missing_city_columns}")

# -----------------------------
# Derived crash variables
# -----------------------------
crashes["year"] = pd.to_numeric(crashes["year"], errors="coerce")
crashes = crashes.loc[crashes["year"].isin(YEARS)].copy()
if crashes.empty:
    raise ValueError("No Dresden crashes found for 2022-2025.")
crashes["year"] = crashes["year"].astype(int)
missing_years = sorted(set(YEARS) - set(crashes["year"]))
if missing_years:
    print(f"WARNING: Missing crash years: {missing_years}")
for field in ["UKATEGORIE", "ULICHTVERH"]:
    if field not in crashes:
        raise ValueError(f"Missing required crash field: {field}")
print("Creating derived crash variables...")

crashes["mode_type"] = crashes.apply(derive_mode_type, axis=1)

if "UKATEGORIE" in crashes.columns:
    crashes["severity_type"] = crashes["UKATEGORIE"].apply(derive_severity_type)
else:
    crashes["severity_type"] = "unknown"

if "ULICHTVERH" in crashes.columns:
    crashes["light_type"] = crashes["ULICHTVERH"].apply(derive_light_type)
else:
    crashes["light_type"] = "unknown"

if "UWOCHENTAG" in crashes.columns:
    # Usually 1-7. We treat 6 and 7 as weekend.
    crashes["weekend"] = crashes["UWOCHENTAG"].apply(
        lambda x: "weekend" if pd.notna(x) and int(x) in [6, 7] else "weekday"
    )
else:
    crashes["weekend"] = "unknown"

# ============================================================
# Load network layers
# ============================================================
print("Loading junction and segment layers...")

junctions = gpd.read_file(JUNCTION_GPKG, layer="junction_areas")
segments = gpd.read_file(SEGMENT_GPKG, layer="Level 4: unified_street_network")
segments["segment_id"] = segments["level4_id"]
segments["edge_id"] = segments["level4_id"].map(lambda value: f"level4_{value}")
source_edges = gpd.read_file(SOURCE_GPKG, layer="Level 3A: debug_relevant_edge_classes")
original_edges = gpd.read_file(ORIGINAL_GPKG, layer="segment_edges")
if not segments.segment_id.is_unique or not junctions.junction_id.is_unique:
    raise ValueError("Network feature IDs must be unique.")
if segments.empty or junctions.empty:
    raise ValueError("Level 4 segments and Level 2 junctions must be nonempty.")
if not junctions.crs or not junctions.crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in junctions.crs.axis_info):
    raise ValueError("Junction CRS must use projected metre coordinates.")
source_edges = source_edges.to_crs(junctions.crs)
original_edges = original_edges.to_crs(junctions.crs)
if not source_edges.source_edge_id.is_unique:
    raise ValueError("Source edge IDs are not unique.")
for frame in [source_edges, original_edges]:
    frame["graph_edge_ids"] = frame[["u", "v", "key"]].astype(str).agg("_".join, axis=1)
    frame["original_osm_way_ids"] = frame.osmid.apply(lambda value: join_ids([value]))
source_lookup = source_edges.set_index("source_edge_id")
source_id_set = set(source_lookup.index)
provenance_rows = []
for _, feature in segments.iterrows():
    ids = [int(value) for value in re.split(r"[;,\s]+", str(feature.source_edge_ids)) if value]
    missing = set(ids) - source_id_set
    if missing:
        raise ValueError(f"Level 4 feature {feature.level4_id} has unknown source edges: {missing}")
    if not ids:
        raise ValueError(f"Level 4 feature {feature.level4_id} has no source edges.")
    for source_id in ids:
        edge = source_lookup.loc[source_id]
        provenance_rows.append({"location_type": "segment", "network_id": feature.level4_id,
                                "source_edge_id": source_id, "graph_edge_id": edge.graph_edge_ids,
                                "original_osm_way_ids": edge.original_osm_way_ids})
# Use the original, untrimmed network so internal junction edges are retained.
junction_links = gpd.sjoin(original_edges[["graph_edge_ids", "original_osm_way_ids", "geometry"]],
                          junctions[["junction_id", "geometry"]], how="inner", predicate="intersects")
for _, edge in junction_links.iterrows():
    provenance_rows.append({"location_type": "intersection", "network_id": edge.junction_id,
                            "source_edge_id": pd.NA, "graph_edge_id": edge.graph_edge_ids,
                            "original_osm_way_ids": edge.original_osm_way_ids})
provenance = pd.DataFrame(provenance_rows).drop_duplicates()
for frame, kind, id_col in [(segments, "segment", "segment_id"), (junctions, "intersection", "junction_id")]:
    links = provenance.loc[provenance.location_type.eq(kind)]
    for output, field in [("graph_edge_ids", "graph_edge_id"), ("original_osm_way_ids", "original_osm_way_ids")]:
        frame[output] = frame[id_col].map(links.groupby("network_id")[field].agg(join_ids)).fillna("")
    if frame.graph_edge_ids.eq("").any():
        raise ValueError(f"Missing original OSM edge provenance for {kind} features.")
if crashes.geometry.isna().any() or crashes.geometry.is_empty.any() or not crashes.geometry.is_valid.all():
    raise ValueError("Crash geometries are missing, empty or invalid; cannot assign every crash.")

if crashes.crs is None:
    raise ValueError("Crash data has no CRS. Check the .prj files.")

if crashes.crs != junctions.crs:
    print("Reprojecting crashes to network CRS...")
    crashes = crashes.to_crs(junctions.crs)

if segments.crs != junctions.crs:
    segments = segments.to_crs(junctions.crs)

# ============================================================
# Classify crashes: intersection vs segment
# ============================================================
print("Classifying crashes as intersection or segment crashes...")

crashes_joined = gpd.sjoin(
    crashes,
    junctions[["junction_id", "graph_edge_ids", "original_osm_way_ids", "geometry"]],
    how="left",
    predicate="intersects",
)

intersection_ties = int(crashes_joined.loc[crashes_joined.duplicated("crash_id", keep=False), "crash_id"].nunique())
crashes_joined = crashes_joined.sort_values(["crash_id", "junction_id"]).drop_duplicates("crash_id").drop(columns="index_right", errors="ignore")
crashes_joined["crash_location_type"] = crashes_joined["junction_id"].apply(
    lambda x: "intersection" if pd.notna(x) else "segment"
)

intersection_crashes = crashes_joined[
    crashes_joined["crash_location_type"] == "intersection"
].copy()

segment_crashes = crashes_joined[
    crashes_joined["crash_location_type"] == "segment"
].copy()

print(f"Intersection crashes: {len(intersection_crashes):,}")
print(f"Segment crashes: {len(segment_crashes):,}")

# ============================================================
# Assign segment crashes to nearest segment
# ============================================================
print("Assigning segment crashes to nearest segment...")

segment_crashes_assigned = gpd.sjoin_nearest(
    segment_crashes.drop(columns=["graph_edge_ids", "original_osm_way_ids"], errors="ignore"),
    segments[["segment_id", "level4_id", "edge_id", "graph_edge_ids", "original_osm_way_ids", "geometry"]],
    how="left",
    max_distance=None,  # Assign every non-intersection crash; flag distant matches below.
    distance_col="distance_to_segment_m",
)

segment_ties = int(segment_crashes_assigned.loc[segment_crashes_assigned.duplicated("crash_id", keep=False), "crash_id"].nunique())
segment_crashes_assigned = segment_crashes_assigned.sort_values(
    ["crash_id", "distance_to_segment_m", "edge_id"]
).drop_duplicates("crash_id").drop(columns="index_right", errors="ignore")
segment_crashes_assigned["beyond_30m"] = segment_crashes_assigned.distance_to_segment_m.gt(MAX_SEGMENT_SNAP_DISTANCE_M)
if segment_crashes_assigned.segment_id.isna().any():
    raise ValueError("Some crashes could not be assigned to Level 4 segments.")
matched_segment_crashes = segment_crashes_assigned[
    segment_crashes_assigned["segment_id"].notna()
].copy()

unmatched_segment_crashes = segment_crashes_assigned[
    segment_crashes_assigned["segment_id"].isna()
].copy()

print(f"Matched segment crashes: {len(matched_segment_crashes):,}")
print(f"Unmatched segment crashes: {len(unmatched_segment_crashes):,}")

if len(segment_crashes) > 0:
    unmatched_share = len(unmatched_segment_crashes) / len(segment_crashes) * 100
    print(f"Unmatched segment crash share: {unmatched_share:.2f}%")

# ============================================================
# Aggregate crashes
# ============================================================
print("Aggregating intersection crashes by junction_id...")
junction_crash_counts = aggregate_counts(
    intersection_crashes,
    group_id_col="junction_id",
    prefix="junction",
)

junctions_with_crashes = junctions.merge(
    junction_crash_counts,
    on="junction_id",
    how="left",
).fillna(0)

print("Aggregating segment crashes by segment_id...")
segment_crash_counts = aggregate_counts(
    matched_segment_crashes,
    group_id_col="segment_id",
    prefix="segment",
)

segments_with_crashes = segments.merge(
    segment_crash_counts,
    on="segment_id",
    how="left",
).fillna(0)

# ============================================================
# Excel export and reconciliation against loaded Unfallatlas data
# ============================================================
assignments = pd.concat([
    intersection_crashes.assign(assignment="intersection"),
    matched_segment_crashes.assign(assignment="segment"),
    unmatched_segment_crashes.assign(assignment="none"),
], ignore_index=True)
assert assignments.graph_edge_ids.notna().all() and assignments.graph_edge_ids.ne("").all(), "Missing OSM provenance"
assert unmatched_segment_crashes.empty, "All crashes must be assigned"
assert assignments.crash_id.is_unique, "Crash assigned more than once"
assert set(assignments.crash_id) == set(crashes.crash_id), "Crash lost during assignment"
print("Reading yearly CSV reference and road-surface conditions...", flush=True)
csv_reference, csv_inventory = load_csv_reference(CRASH_CSV_DIR)
missing_csv_years = sorted(set(YEARS) - set(csv_reference.year))
if missing_csv_years:
    print(f"WARNING: Missing CSV reference years: {missing_csv_years}")
summary_rows, comparison_rows = [], []
for year in [*YEARS, "2022-2025"]:
    original = crashes if isinstance(year, str) else crashes.loc[crashes.year.eq(year)]
    reference = csv_reference if isinstance(year, str) else csv_reference.loc[csv_reference.year.eq(year)]
    assigned = assignments if isinstance(year, str) else assignments.loc[assignments.year.eq(year)]
    row = {"year": year, "shapefile_input": len(original), "csv_reference": len(reference),
           "csv_minus_shapefile": len(reference) - len(original)}
    for kind in ["intersection", "segment", "none"]:
        row[kind] = int(assigned.assignment.eq(kind).sum())
        row[f"{kind}_pct"] = 100 * row[kind] / len(original) if len(original) else 0
    row["assignment_reconciliation_difference"] = len(original) - sum(row[kind] for kind in ["intersection", "segment", "none"])
    row["csv_minus_assigned"] = len(reference) - row["intersection"] - row["segment"]
    row["csv_minus_all_outcomes"] = len(reference) - sum(row[kind] for kind in ["intersection", "segment", "none"])
    row["segment_beyond_30m"] = int(assigned.loc[assigned.assignment.eq("segment"), "beyond_30m"].fillna(False).sum())
    summary_rows.append(row)
    fields = {"UKATEGORIE": [1, 2, 3], "ULICHTVERH": [0, 1, 2], **{col: [1] for col in MODE_COLUMNS}}
    for field, codes in fields.items():
        if field not in original:
            continue
        codes = [*codes, "unknown"] if field not in MODE_COLUMNS else codes
        for code in codes:
            raw_values = pd.to_numeric(original[field], errors="coerce")
            values = pd.to_numeric(assigned[field], errors="coerce")
            valid_codes = [c for c in codes if c != "unknown"]
            raw_mask = ~raw_values.isin(valid_codes) if code == "unknown" else raw_values.eq(code)
            mask = ~values.isin(valid_codes) if code == "unknown" else values.eq(code)
            if field in reference:
                csv_values = pd.to_numeric(reference[field], errors="coerce")
                csv_mask = ~csv_values.isin(valid_codes) if code == "unknown" else csv_values.eq(code)
                csv_count = int(csv_mask.sum())
            else:
                csv_count = pd.NA
            detail = {"year": year, "field": field, "code": code,
                      "shapefile_input": int(raw_mask.sum()), "csv_reference": csv_count,
                      "csv_minus_shapefile": csv_count - int(raw_mask.sum())}
            for kind in ["intersection", "segment", "none"]:
                detail[kind] = int((mask & assigned.assignment.eq(kind)).sum())
            detail["assignment_reconciliation_difference"] = detail["shapefile_input"] - sum(detail[kind] for kind in ["intersection", "segment", "none"])
            detail["csv_minus_assigned"] = csv_count - detail["intersection"] - detail["segment"]
            detail["csv_minus_all_outcomes"] = csv_count - sum(detail[kind] for kind in ["intersection", "segment", "none"])
            comparison_rows.append(detail)
summary = pd.DataFrame(summary_rows)
# Attach reference-only attributes without changing the spatial assignment.
# IstStrassenzustand is unavailable in some shapefiles, so recover it from the yearly CSV.
if "IstStrassenzustand" not in assignments and "IstStrassenzustand" in csv_reference:
    crash_key = next((col for col in ["UIDENTSTLAE", "UIDENTSTLA"] if col in assignments), None)
    ref_key = next((col for col in ["UIDENTSTLAE", "UIDENTSTLA"] if col in csv_reference), None)
    if crash_key and ref_key:
        reference = csv_reference[["year", ref_key, "IstStrassenzustand"]].copy()
        reference["_reference_id"] = reference[ref_key].astype("string").str.strip()
        if reference.duplicated(["year", "_reference_id"]).any():
            raise ValueError("Duplicate year/crash IDs in road surface reference.")
        assignments["_reference_id"] = assignments[crash_key].astype("string").str.strip()
        assignments = assignments.merge(
            reference[["year", "_reference_id", "IstStrassenzustand"]],
            on=["year", "_reference_id"], how="left", validate="many_to_one",
        ).drop(columns="_reference_id")

print("Checking Level 0-4 lineage and building feature inventory...", flush=True)
inventory = build_network_inventory(segments, junctions, source_edges, original_edges)
network_counts = export_counts(assignments, inventory)
OUTPUT_CSV = OUTPUT_DIR / f"{CITY.lower()}_network_crash_counts_2022_2025.csv"
OUTPUT_EXCEL = OUTPUT_DIR / f"{CITY.lower()}_crash_counts_2022_2025.xlsx"

# Individual-crash sheets preserve all source attributes and assignment metadata.
crash_details = pd.DataFrame(assignments.drop(columns="geometry"))
crash_details["crash_geometry_wkt"] = assignments.geometry.to_wkt().to_numpy()
crash_details["crash_geometry_crs"] = str(assignments.crs)
crash_details["location_type"] = crash_details.assignment
crash_details["network_id"] = crash_details.segment_id.where(
    crash_details.assignment.eq("segment"), crash_details.junction_id
)
detail_inventory = inventory.drop(columns=["level4_id", "junction_id"])
crash_details = crash_details.merge(
    detail_inventory, on=["location_type", "network_id"], how="left", validate="many_to_one",
    suffixes=("", "_network"),
)
tables = {
    "network_counts": network_counts,
    "segment_crashes": crash_details.loc[crash_details.assignment.eq("segment")],
    "intersection_crashes": crash_details.loc[crash_details.assignment.eq("intersection")],
    "unassigned_crashes": crash_details.loc[crash_details.assignment.eq("none")],
}
try:
    import openpyxl
except ImportError as exc:
    raise ImportError("Excel export needs openpyxl: install osm_crash_network/requirements.txt.") from exc
# Reject oversized cells before writing, since Excel otherwise truncates text.
for sheet_name, table in tables.items():
    for column in table.select_dtypes(include=["object", "string"]).columns:
        if table[column].astype("string").str.len().gt(32767).any():
            raise ValueError(f"{sheet_name}.{column} exceeds Excel cell capacity.")
network_counts.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    for sheet_name, table in tables.items():
        table.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cells in worksheet.columns:
            worksheet.column_dimensions[cells[0].column_letter].width = min(60, max(14, len(str(cells[0].value)) + 2))
print(f"Saved combined CSV: {OUTPUT_CSV}")
print("Crash assignment statistics versus Unfallatlas CSV reference:")
print(summary.to_string(index=False))
print(f"Ties resolved: intersections={intersection_ties:,}, segments={segment_ties:,}")
print(f"Saved Excel workbook: {OUTPUT_EXCEL}")

# ============================================================
# Save outputs
# ============================================================
print("Saving outputs...")
# The complete classified layer includes the final Level 4 assignment and provenance.
assignments.to_file(OUTPUT_GPKG, layer="all_classified_crashes", driver="GPKG")

crashes.to_file(
    OUTPUT_GPKG,
    layer="dresden_crashes_only",
    driver="GPKG",
)



intersection_crashes.to_file(
    OUTPUT_GPKG,
    layer="intersection_crashes",
    driver="GPKG",
)

matched_segment_crashes.to_file(
    OUTPUT_GPKG,
    layer="segment_crashes_assigned",
    driver="GPKG",
)

unmatched_segment_crashes.to_file(
    OUTPUT_GPKG,
    layer="segment_crashes_unmatched",
    driver="GPKG",
)

junctions_with_crashes.to_file(
    OUTPUT_GPKG,
    layer="junctions_with_crash_counts",
    driver="GPKG",
)

segments_with_crashes.to_file(
    OUTPUT_GPKG,
    layer="segments_with_crash_counts",
    driver="GPKG",
)

print("=" * 60)
print("Done.")
print(f"Saved GeoPackage:")
print(f"  {OUTPUT_GPKG}")
print("=" * 60)