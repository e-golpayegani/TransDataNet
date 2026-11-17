"""Level 5: Important junction approaches (signalized and primary/secondary).

Run: python osm_crash_network/src/06_create_important_junction_approaches.py
OSM evidence describes mapped signals/crossings, not verified absence in the field.
"""
from pathlib import Path
import argparse
import json
import re
import os
import subprocess

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import substring

DATA_ROOT = Path(r"Z:\_Public\Promotionen\Golpayegani\Paper3\03_OSM_Network_Simplification\GIS_Data")
CITY = "Dresden"
# This rebuilt pair matches the current 4,547-junction Level 2 dataset.
INPUT_NETWORK_SUFFIX = "_rebuilt_20261030"
CROSSWALK_EXTENSION_M = 50.0
CENTROID_RADIUS_M = 75.0
CROSSING_EDGE_TOLERANCE_M = 3.0
MIN_PART_M = 0.05
MAJOR_HIGHWAYS = {"primary", "secondary"}


def tokens(value):
    if isinstance(value, (list, tuple, set)):
        return set().union(*(tokens(item) for item in value)) if value else set()
    if value is None or pd.isna(value):
        return set()
    return {part.strip(" []'\"") for part in re.split(r"[;,]", str(value)) if part.strip(" []'\"")}


def ids_text(values):
    return ";".join(sorted(set().union(*(tokens(value) for value in values))))


def node_id(value):
    return str(int(value)) if isinstance(value, (int, float)) and pd.notna(value) else str(value)


def points_in(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [point for part in geometry.geoms for point in points_in(part)]
    return []


def required(frame, fields, label):
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing fields {sorted(missing)}")
    if frame.empty or frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ValueError(f"{label}: empty dataset or invalid geometry")


def orient(line, start_polygon, end_polygon):
    """Orient by distance to the actual junction polygons, not endpoint metadata order."""
    first, last = Point(line.coords[0]), Point(line.coords[-1])
    forward = (first.distance(start_polygon) if start_polygon is not None else 0) + (last.distance(end_polygon) if end_polygon is not None else 0)
    reverse = (last.distance(start_polygon) if start_polygon is not None else 0) + (first.distance(end_polygon) if end_polygon is not None else 0)
    return LineString(list(line.coords)[::-1]) if reverse < forward else line


def classify_junctions(junctions, source, nodes):
    junctions = junctions.copy()
    signals = nodes.loc[nodes.highway.apply(lambda value: "traffic_signals" in tokens(value))].copy()
    crossings = nodes.loc[nodes.highway.apply(lambda value: "crossing" in tokens(value))].copy()
    signal_links = gpd.sjoin(signals[["osm_node_id", "geometry"]], junctions[["junction_id", "geometry"]], how="inner", predicate="intersects")
    counts = signal_links.groupby("junction_id").osm_node_id.nunique()
    junctions["signal_node_count"] = junctions.junction_id.map(counts).fillna(0).astype(int)
    junctions["is_signalized"] = junctions.signal_node_count.gt(0)
    junctions["signal_osm_node_ids"] = junctions.junction_id.map(signal_links.groupby("junction_id").osm_node_id.agg(ids_text)).fillna("")
    # Contracted endpoints retain the Level 2 topology and exclude internal edges.
    leg_rows = []
    for edge in source.itertuples():
        if edge.cu == edge.cv:
            continue
        for endpoint in [edge.cu, edge.cv]:
            if str(endpoint).startswith("JUNCTION_"):
                jid = int(float(str(endpoint).split("_", 1)[1]))
                leg_rows.append({"junction_id": jid, "source_edge_id": edge.source_edge_id,
                                 "highway": edge.highway, "major": bool(tokens(edge.highway) & MAJOR_HIGHWAYS)})
    legs = pd.DataFrame(leg_rows, columns=["junction_id", "source_edge_id", "highway", "major"])
    major = legs.loc[legs.major]
    junctions["has_primary_secondary_leg"] = junctions.junction_id.isin(major.junction_id)
    junctions["incident_highway_types"] = junctions.junction_id.map(legs.groupby("junction_id").highway.agg(ids_text)).fillna("")
    return junctions, signals, crossings


def crossing_positions(segments, source, crossings):
    """Match raw crossing nodes to original source roads, then project to Level 4."""
    mapping = []
    for row in segments.itertuples():
        for source_id in tokens(row.source_edge_ids):
            mapping.append({"level4_id": row.level4_id, "source_edge_id": int(source_id)})
    mapping = pd.DataFrame(mapping).drop_duplicates()
    unknown = set(mapping.source_edge_id) - set(source.source_edge_id)
    if unknown:
        raise ValueError(f"Level 4 source lineage has unknown edges: {sorted(unknown)[:10]}")
    if crossings.empty:
        return {}, gpd.GeoDataFrame(columns=["osm_node_id", "source_edge_id", "level4_id", "geometry"], geometry="geometry", crs=segments.crs)
    links = gpd.sjoin(crossings[["osm_node_id", "geometry"]], source[["source_edge_id", "geometry"]],
                      how="inner", predicate="dwithin", distance=CROSSING_EDGE_TOLERANCE_M)
    links = links.drop(columns="index_right").merge(mapping, on="source_edge_id", how="inner")
    links = links.drop_duplicates(["osm_node_id", "source_edge_id", "level4_id"])
    geometries = segments.set_index("level4_id").geometry
    links["position_m"] = [geometries.loc[lid].project(point) for lid, point in zip(links.level4_id, links.geometry)]
    links["distance_to_centerline_m"] = [geometries.loc[lid].distance(point) for lid, point in zip(links.level4_id, links.geometry)]
    lookup = {lid: group.drop_duplicates("osm_node_id").to_dict("records") for lid, group in links.groupby("level4_id")}
    return lookup, links


def leg_cut(line, polygon, crossing_candidates, from_start, other_junction_exists):
    """Return an outward cut distance; short legs remain wholly approach parts."""
    length = line.length
    oriented = line if from_start else LineString(list(line.coords)[::-1])
    boundary_positions = sorted(oriented.project(point) for point in points_in(oriented.intersection(polygon.boundary)))
    # Use the first exit from this polygon in the outward direction.
    exit_distance = 0.0
    for position in boundary_positions:
        if not polygon.covers(oriented.interpolate(min(length, position + MIN_PART_M))):
            exit_distance = position
            break
    eligible = []
    for crossing in crossing_candidates:
        outward = crossing["position_m"] if from_start else length - crossing["position_m"]
        if other_junction_exists and outward > length / 2 + MIN_PART_M:
            continue
        if outward < exit_distance - MIN_PART_M and not polygon.covers(crossing["geometry"]):
            continue
        eligible.append((outward, str(crossing["osm_node_id"]), crossing))
    centroid_adjusted = False
    if eligible:
        distance, crossing_id, evidence = min(eligible, key=lambda item: (item[0], item[1]))
        target = max(distance + CROSSWALK_EXTENSION_M, exit_distance + MIN_PART_M)
        rule = "crossing_plus_50m"
    else:
        crossing_id, distance, evidence = "", None, None
        radius = polygon.centroid.buffer(CENTROID_RADIUS_M)
        exits = sorted(oriented.project(point) for point in points_in(oriented.intersection(radius.boundary)))
        exits = [position for position in exits if position >= exit_distance - MIN_PART_M and not radius.covers(oriented.interpolate(min(length, position + MIN_PART_M)))]
        if exits:
            target = exits[0]
            rule = "centroid_75m_radius"
        elif radius.covers(oriented.interpolate(length)):
            target = length
            rule = "centroid_75m_short_leg"
        else:
            # Large/irregular junctions may already extend beyond the centroid radius.
            target = exit_distance + CROSSWALK_EXTENSION_M
            rule = "large_junction_boundary_plus_50m"
            centroid_adjusted = True
    actual = min(length, max(0.0, target))
    return {"outward_cut_m": actual, "requested_outward_m": target, "rule": rule,
            "crossing_osm_node_id": crossing_id, "crossing_outward_m": distance,
            "junction_exit_m": exit_distance, "whole_leg_approach": actual >= length - MIN_PART_M,
            "centroid_fallback_adjusted": centroid_adjusted,
            "crossing_distance_to_centerline_m": evidence["distance_to_centerline_m"] if evidence else None}


def split_variant(name, flag, junctions, segments, crossing_lookup):
    junctions = junctions.copy()
    junctions["is_important"] = junctions[flag].astype(bool)
    junctions["importance_name"] = junctions.is_important.map({True: f"important_{name}", False: "other_junction"})
    areas = junctions.set_index("junction_id")
    important = set(junctions.loc[junctions.is_important, "junction_id"])
    parts, leg_rows, cut_rows = [], [], []
    for row in segments.to_dict("records"):
        line = row["geometry"]
        length = line.length
        sid = row["level4_id"]
        proposals = []
        for side, field in [("start", "start_junction_id"), ("end", "end_junction_id")]:
            jid = row[field]
            if pd.isna(jid) or jid not in important:
                continue
            other = row["end_junction_id" if side == "start" else "start_junction_id"]
            cut = leg_cut(line, areas.loc[jid].geometry, crossing_lookup.get(sid, []), side == "start", pd.notna(other))
            cut.update({"variant": name, "level4_id": sid, "junction_id": int(jid), "side": side})
            cut["position_m"] = cut["outward_cut_m"] if side == "start" else length - cut["outward_cut_m"]
            proposals.append(cut)
        # Short-parent rule has priority: do not split or relabel any part as an approach.
        short_parent = any(proposal["whole_leg_approach"] for proposal in proposals)
        active_proposals = [] if short_parent else proposals
        cuts = sorted({0.0, length, *(p["position_m"] for p in active_proposals if MIN_PART_M < p["position_m"] < length - MIN_PART_M)})
        # Merge nearly coincident cuts to avoid tiny artificial pieces.
        clean = [cuts[0]]
        for cut in cuts[1:]:
            if cut - clean[-1] < MIN_PART_M:
                if cut == length:
                    clean[-1] = length
            else:
                clean.append(cut)
        if len(clean) == 1:
            clean = [0.0, length]
        start_target = next((p["position_m"] for p in active_proposals if p["side"] == "start"), None)
        end_target = next((p["position_m"] for p in active_proposals if p["side"] == "end"), None)
        overlap = start_target is not None and end_target is not None and start_target > end_target + MIN_PART_M
        for proposal in proposals:
            proposal["skipped_short_parent"] = short_parent
            proposal["overlapping_approaches"] = overlap
            proposal["actual_interior_cut"] = any(abs(proposal["position_m"] - pos) < MIN_PART_M for pos in clean[1:-1])
            leg_rows.append(proposal.copy())
            if proposal["actual_interior_cut"]:
                cut_rows.append({**proposal, "geometry": line.interpolate(proposal["position_m"])})
        for part_number, (left, right) in enumerate(zip(clean, clean[1:]), start=1):
            midpoint = (left + right) / 2
            owners = [p for p in active_proposals if (p["side"] == "start" and midpoint <= p["position_m"]) or (p["side"] == "end" and midpoint >= p["position_m"])]
            record = {**row, "geometry": line if short_parent else substring(line, left, right), "variant": name,
                      "level5_id": f"{name}_{sid}_{part_number}", "part_number": part_number,
                      "parent_level4_id": sid, "from_m": left, "to_m": right,
                      "parent_length_m": length, "length_m": right - left,
                      "is_approach": bool(owners), "part_type": "unchanged_short_leg" if short_parent else "important_approach" if owners else "other_segment",
                      "unchanged_short_leg": short_parent,
                      "important_junction_ids": ";".join(str(p["junction_id"]) for p in owners),
                      "cut_rules": ";".join(sorted({p["rule"] for p in owners})),
                      "crossing_osm_node_ids": ";".join(sorted({p["crossing_osm_node_id"] for p in owners if p["crossing_osm_node_id"]})),
                      "overlapping_approaches": overlap, "approach_to_both_ends": len(owners) > 1,
                      "parent_was_split": len(clean) > 2}
            # Endpoint IDs now describe the part, while original topology remains explicit.
            record["parent_start_junction_id"] = row["start_junction_id"]
            record["parent_end_junction_id"] = row["end_junction_id"]
            record["start_junction_id"] = row["start_junction_id"] if left == 0 else None
            record["end_junction_id"] = row["end_junction_id"] if right == length else None
            record["start_node_id"] = f"j{int(row['start_junction_id'])}" if left == 0 and pd.notna(row["start_junction_id"]) else f"l4_{sid}_{left:.6f}"
            record["end_node_id"] = f"j{int(row['end_junction_id'])}" if right == length and pd.notna(row["end_junction_id"]) else f"l4_{sid}_{right:.6f}"
            parts.append(record)
    parts = gpd.GeoDataFrame(parts, geometry="geometry", crs=segments.crs)
    legs = pd.DataFrame(leg_rows)
    cuts = gpd.GeoDataFrame(cut_rows, geometry="geometry", crs=segments.crs) if cut_rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=segments.crs)
    # Each original street is partitioned exactly once; no geometry is lost or duplicated.
    lengths = parts.groupby("parent_level4_id").length_m.sum()
    expected = segments.set_index("level4_id").geometry.length
    if set(lengths.index) != set(expected.index) or (lengths.subtract(expected).abs() > 1e-6).any():
        raise AssertionError("Level 5 split does not conserve parent lengths")
    if not parts.level5_id.is_unique or (parts.geometry.length - parts.length_m).abs().gt(1e-6).any():
        raise AssertionError("Invalid Level 5 geometry or identifiers")
    summary = {"variant": name, "junctions_total": len(junctions), "junctions_important": int(junctions.is_important.sum()),
               "junctions_other": int((~junctions.is_important).sum()), "segments_level4": len(segments),
               "segments_level5": len(parts), "approach_parts": int(parts.is_approach.sum()),
               "other_parts": int(parts.part_type.eq("other_segment").sum()),
               "unchanged_short_legs": int(parts.unchanged_short_leg.sum()), "parents_split": int(parts.loc[parts.parent_was_split, "parent_level4_id"].nunique()),
               "important_legs": len(legs), "legs_with_crossing": int(legs.rule.eq("crossing_plus_50m").sum()) if len(legs) else 0,
               "legs_without_mapped_crossing": int(legs.crossing_osm_node_id.eq("").sum()) if len(legs) else 0,
               "short_requested_approaches": int(legs.whole_leg_approach.sum()) if len(legs) else 0,
               "legs_skipped_for_short_parent": int(legs.skipped_short_parent.sum()) if len(legs) else 0,
               "overlap_parents": int(parts.loc[parts.overlapping_approaches, "parent_level4_id"].nunique()),
               "centroid_fallback_adjustments": int(legs.centroid_fallback_adjusted.sum()) if len(legs) else 0,
               "important_junctions_without_level4_leg": len(important - set(legs.junction_id)) if len(legs) else len(important),
               "original_length_m": float(expected.sum()), "level5_length_m": float(parts.length_m.sum())}
    return junctions, parts, legs, cuts, summary


def save_qgis_project(data_root, city):
    """Run QGIS Python to retain existing project layers and add styled Level 5 groups."""
    launchers = sorted(Path("C:/Program Files").glob("QGIS */bin/python-qgis*.bat"), reverse=True)
    if not launchers:
        raise RuntimeError("QGIS Python launcher not found. Data outputs are saved, but the combined QGIS project could not be created.")
    # Prefer the installed 3.44 release used for the project.
    launcher = next((path for path in launchers if "3.44" in str(path)), launchers[0])
    subprocess.run([str(launcher), str(Path(__file__).with_name("qgis_level5_project.py").resolve()),
                    "--data-root", str(Path(data_root).resolve()), "--city", city],
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, check=True)


def build(data_root=DATA_ROOT, city=CITY, network_suffix=INPUT_NETWORK_SUFFIX):
    city_dir = Path(data_root) / city
    processed, outputs = city_dir / "processed", city_dir / "outputs"
    segments = gpd.read_file(processed / f"{city.lower()}_unified_street_network{network_suffix}.gpkg", layer="Level 4: unified_street_network")
    junctions = gpd.read_file(processed / f"{city.lower()}_junction_areas.gpkg", layer="junction_areas")
    source = gpd.read_file(processed / f"{city.lower()}_j2j_detailed_network{network_suffix}.gpkg", layer="Level 3A: debug_relevant_edge_classes")
    nodes = gpd.read_file(city_dir / "raw" / f"{city.lower()}_raw_osm_network.gpkg", layer="raw_nodes")
    required(segments, ["level4_id", "source_edge_ids", "start_junction_id", "end_junction_id"], "Level 4")
    required(junctions, ["junction_id"], "Level 2")
    required(source, ["source_edge_id", "cu", "cv", "highway", "osmid"], "Level 3 source lineage")
    required(nodes, ["osmid", "highway"], "Raw OSM nodes")
    print(f"Input network suffix: {network_suffix or '(default files)'}; junctions={len(junctions):,}; streets={len(segments):,}", flush=True)
    crs = junctions.crs
    if crs is None or not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info):
        raise ValueError("A projected metre CRS is required")
    segments, source, nodes = [frame.to_crs(crs) for frame in [segments, source, nodes]]
    for frame, key in [(segments, "level4_id"), (junctions, "junction_id"), (source, "source_edge_id")]:
        if not frame[key].is_unique:
            raise ValueError(f"Duplicate identifiers: {key}")
    if not segments.geometry.geom_type.eq("LineString").all() or segments.geometry.length.le(0).any():
        raise ValueError("Every Level 4 feature must be a nonempty LineString")
    areas = junctions.set_index("junction_id").geometry
    unknown_junctions = set(pd.concat([segments.start_junction_id, segments.end_junction_id]).dropna()) - set(areas.index)
    if unknown_junctions:
        raise ValueError(f"Unknown Level 4 endpoint junctions: {unknown_junctions}")
    segments.geometry = [orient(row.geometry, areas.loc[row.start_junction_id] if pd.notna(row.start_junction_id) else None,
                                areas.loc[row.end_junction_id] if pd.notna(row.end_junction_id) else None) for row in segments.itertuples()]
    # Keep inherited Level 4 lineage on each child. It refers to the parent, not exclusively to this child.
    source_lookup = source.set_index("source_edge_id")
    segments["parent_original_osm_way_ids"] = [ids_text(source_lookup.loc[[int(value) for value in tokens(row.source_edge_ids)], "osmid"]) for row in segments.itertuples()]
    # Validate source topology against the chosen Level 2 polygons, including IDs.
    bad_source_endpoints = []
    for edge in source.itertuples():
        endpoints = [Point(edge.geometry.coords[0]), Point(edge.geometry.coords[-1])]
        for endpoint in {str(edge.cu), str(edge.cv)}:
            if endpoint.startswith("JUNCTION_"):
                jid = int(float(endpoint.split("_", 1)[1]))
                if jid not in areas.index or min(point.distance(areas.loc[jid]) for point in endpoints) > 0.1:
                    bad_source_endpoints.append((edge.source_edge_id, jid))
    if bad_source_endpoints:
        raise ValueError(f"Source network and Level 2 junction geometries do not match: {bad_source_endpoints[:10]}")
    nodes["osm_node_id"] = nodes.osmid.map(node_id)
    print("Classifying junctions from raw OSM signals and incident road classes...", flush=True)
    junctions, signals, crossings = classify_junctions(junctions, source, nodes)
    print("Matching crossing nodes to original source roads...", flush=True)
    crossing_lookup, crossing_links = crossing_positions(segments, source, crossings)
    results = [split_variant(name, flag, junctions, segments, crossing_lookup) for name, flag in
               [("signalized", "is_signalized"), ("primary_secondary", "has_primary_secondary_leg")]]
    outputs.mkdir(parents=True, exist_ok=True)
    output = processed / f"{city.lower()}_level5_networks.gpkg"
    # Build a new file, then replace the previous Level 5 output only after success.
    pending = output.with_name(output.stem + "_pending.gpkg")
    if pending.exists():
        raise FileExistsError(f"Previous incomplete Level 5 output exists: {pending}. Inspect or rename it before retrying.")
    for name, result in zip(["signalized", "primary_secondary"], results):
        junction_frame, parts, legs, cuts, summary = result
        for suffix, frame in [("junctions", junction_frame), ("segments", parts), ("approach_parts", parts.loc[parts.is_approach]),
                              ("other_parts", parts.loc[parts.part_type.eq("other_segment")]),
                              ("unchanged_short_legs", parts.loc[parts.unchanged_short_leg]), ("cut_points", cuts)]:
            frame.to_file(pending, layer=f"{name}_{suffix}", driver="GPKG")
        legs.to_csv(outputs / f"{city.lower()}_level5_{name}_leg_checks.csv", index=False)
    signals.to_file(pending, layer="osm_signal_evidence", driver="GPKG")
    crossings.to_file(pending, layer="osm_crossing_evidence", driver="GPKG")
    crossing_links.to_file(pending, layer="crossing_source_matches", driver="GPKG")
    pending.replace(output)
    summary = pd.DataFrame([result[-1] for result in results])
    summary.to_csv(outputs / f"{city.lower()}_level5_statistics.csv", index=False)
    comparison = junctions[["junction_id", "is_signalized", "has_primary_secondary_leg"]].copy()
    with pd.ExcelWriter(outputs / f"{city.lower()}_level5_statistics.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="variant_statistics", index=False)
        comparison.to_excel(writer, sheet_name="junction_comparison", index=False)
        for name, result in zip(["signalized", "primary_secondary"], results):
            result[2].to_excel(writer, sheet_name=f"{name}_legs", index=False)
    notes = {"input_network": "Level 4 segments and Level 2 junction polygons", "network_suffix": network_suffix, "signals": "raw highway=traffic_signals nodes inside/intersecting junction polygons",
             "major_roads": "at least one incident external source-road leg tagged primary or secondary; link types excluded",
             "crossings": "raw highway=crossing nodes within 3 m of a source edge represented in Level 4; nearest crossing on this endpoint's half of the leg",
             "crossing_cut": "50 m along the Level 4 line beyond the crossing, away from this junction",
             "no_crossing_cut": "first outward intersection with the 75 m centroid-radius circle; oversized junction fallback is boundary exit plus 50 m",
             "short_legs": "if either requested approach reaches the opposite endpoint, keep the entire parent unchanged, with no cut or approach classification",
             "two_important_ends": "two approach parts and a middle part when cuts do not overlap; overlapping zones are partitioned without duplicate geometry",
             "lineage": "inherited source IDs and parent_original_osm_way_ids describe the entire parent; children retain parent_level4_id",
             "absence": "no mapped crossing or signal is not proof of real-world absence",
             "parameters": {"crosswalk_extension_m": CROSSWALK_EXTENSION_M, "centroid_radius_m": CENTROID_RADIUS_M, "crossing_source_tolerance_m": CROSSING_EDGE_TOLERANCE_M}}
    (outputs / f"{city.lower()}_level5_method.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved QGIS layers: {output}", flush=True)
    save_qgis_project(data_root, city)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--city", default=CITY)
    parser.add_argument("--network-suffix", default=INPUT_NETWORK_SUFFIX, help="Matched Level 3/4 filename suffix; use an empty string for default filenames")
    args = parser.parse_args()
    build(args.data_root, args.city, args.network_suffix)
