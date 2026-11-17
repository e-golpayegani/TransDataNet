"""Write the combined Level 0-5 QGIS project using the installed QGIS Python."""
from pathlib import Path
import argparse
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def write_project(data_root, city, output_date=None):
    from qgis.core import (QgsApplication, QgsProject, QgsVectorLayer, QgsLineSymbol,
                           QgsFillSymbol, QgsMarkerSymbol, QgsRendererCategory, QgsCategorizedSymbolRenderer,
                           QgsSingleSymbolRenderer, QgsCentroidFillSymbolLayer, QgsRenderContext, QgsExpressionContextUtils)
    from qgis.PyQt.QtCore import QDateTime, QTimeZone
    from qgis.PyQt.QtGui import QColor
    application = QgsApplication([], False)
    application.initQgis()
    try:
        root = Path(data_root) / city
        outputs = root / "outputs"
        source = outputs / f"{city.lower()}_network_levels_0_to_4_20261030.qgz"
        network = root / "processed" / f"{city.lower()}_level5_networks.gpkg"
        if not source.exists() or not network.exists():
            raise FileNotFoundError(f"Required QGIS project or Level 5 dataset is missing: {source}, {network}")
        stamp = output_date or QDateTime.currentDateTimeUtc().toTimeZone(QTimeZone(b"Europe/Berlin")).toString("yyyyMMdd")
        if len(stamp) != 8 or not stamp.isdigit():
            raise ValueError("Output date must be YYYYMMDD")
        destination = outputs / f"{city.lower()}_network_levels_0_to_5_{stamp}.qgz"
        project = QgsProject.instance()
        if not project.read(str(source)):
            raise RuntimeError(f"Cannot read QGIS project: {source}")
        original_ids = set(project.mapLayers())
        tree = project.layerTreeRoot()
        # Keep earlier layers available; show one Level 5 comparison variant on opening.
        for child in tree.children():
            child.setItemVisibilityChecked(child.name() == "OpenStreetMap")
        level5 = tree.insertGroup(0, "Level 5 - Important junction approaches")
        specifications = [
            ("junctions", "Important junctions only", "important_only", True),
            ("junctions", "All junction areas - important / other", "polygon", False),
            ("approach_parts", "Approach parts near important junctions", "approach", True),
            ("other_parts", "Remaining street parts", "other", True),
            ("unchanged_short_legs", "Short legs - unchanged", "short", True),
            ("cut_points", "Split locations", "point", False),
            ("segments", "Complete Level 5 street network", "complete", False),
        ]
        added = []
        for variant, title, color in [("signalized", "A - Signalized junction approaches", "#d95f02"),
                                      ("primary_secondary", "B - Primary / secondary junction approaches", "#1f78b4")]:
            group = level5.addGroup(title)
            for suffix, label, style, visible in specifications:
                layer = QgsVectorLayer(f"{network}|layername={variant}_{suffix}", f"Level 5 - {label}", "ogr")
                if not layer.isValid():
                    raise RuntimeError(f"Invalid Level 5 layer: {variant}_{suffix}")
                layer.setAbstract(f"Level 5: {title}. Parent Level 4 IDs and split-method fields are retained. See the Level 5 statistics workbook and leg-check CSVs.")
                rgba = QColor(color)
                fill_color = f"{rgba.red()},{rgba.green()},{rgba.blue()},65"
                if style == "important_only":
                    if not layer.setSubsetString('"is_important" = 1'):
                        raise RuntimeError("Cannot filter important junctions")
                    layer.setName(f"Level 5 - Important junctions only ({layer.featureCount():,})")
                    symbol = QgsFillSymbol.createSimple({"color": fill_color, "outline_color": color, "outline_width": "0.8"})
                    centroid = QgsCentroidFillSymbolLayer.create({"point_on_surface": "1", "point_on_all_parts": "0"})
                    centroid.setSubSymbol(QgsMarkerSymbol.createSimple({"name": "circle", "color": color, "outline_color": "white", "outline_width": "0.35", "size": "3"}))
                    symbol.appendSymbolLayer(centroid)
                    layer.setRenderer(QgsSingleSymbolRenderer(symbol))
                elif style == "polygon":
                    important = QgsFillSymbol.createSimple({"color": fill_color, "outline_color": color, "outline_width": "0.6"})
                    other = QgsFillSymbol.createSimple({"color": "210,210,210,35", "outline_color": "150,150,150,100", "outline_width": "0.15"})
                    layer.setRenderer(QgsCategorizedSymbolRenderer('CASE WHEN "is_important" THEN 1 ELSE 0 END', [QgsRendererCategory(1, important, "Important junction"), QgsRendererCategory(0, other, "Other junction")]))
                elif style == "point":
                    layer.renderer().setSymbol(QgsMarkerSymbol.createSimple({"name": "circle", "color": "#e7298a", "size": "2"}))
                else:
                    layer.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": color if style in ["approach", "complete"] else "#999999", "line_width": "1.0" if style == "approach" else "0.35"}))
                if style == "polygon":
                    context = QgsRenderContext()
                    context.expressionContext().appendScope(QgsExpressionContextUtils.layerScope(layer))
                    context.expressionContext().setFields(layer.fields())
                    renderer = layer.renderer()
                    renderer.startRender(context, layer.fields())
                    try:
                        for feature in layer.getFeatures():
                            context.expressionContext().setFeature(feature)
                            if renderer.symbolForFeature(feature, context) is None:
                                raise RuntimeError("Junction category renderer omitted a feature")
                    finally:
                        renderer.stopRender(context)
                project.addMapLayer(layer, False)
                node = group.addLayer(layer)
                node.setItemVisibilityChecked(visible)
                added.append(layer.id())
        evidence = level5.addGroup("OSM evidence - signals and crossings")
        for table, title in [("osm_signal_evidence", "Raw OSM traffic signals"), ("osm_crossing_evidence", "Raw OSM crossings"), ("crossing_source_matches", "Crossings matched to source roads")]:
            layer = QgsVectorLayer(f"{network}|layername={table}", title, "ogr")
            if not layer.isValid():
                raise RuntimeError(f"Invalid OSM evidence layer: {table}")
            project.addMapLayer(layer, False)
            evidence.addLayer(layer).setItemVisibilityChecked(False)
            added.append(layer.id())
        evidence.setItemVisibilityChecked(False)
        level5.children()[1].setItemVisibilityChecked(False)
        level5.children()[0].setItemVisibilityChecked(True)
        crash_file = outputs / "crash_analysis_level5_signalized" / f"{city.lower()}_classified_and_aggregated_crashes.gpkg"
        if crash_file.exists():
            crash_group = tree.insertGroup(1, "Crash analysis - Level 5 signalized (primary 30m / others 10m)")
            for table, label, color, visible in [
                ("segment_crashes_unmatched", "Unassigned crashes", "#e31a1c", True),
                ("intersection_crashes", "Intersection crashes", "#33a02c", True),
                ("segment_crashes_assigned", "Assigned segment crashes", "#1f78b4", True),
                ("all_classified_crashes", "All crashes with assignment status", "#6a3d9a", False),
                ("dresden_crashes_only", "Original Dresden crash points", "#999999", False),
                ("junctions_with_crash_counts", "Junction areas with crash counts", "#33a02c", False),
                ("segments_with_crash_counts", "Level 5 segments with crash counts", "#1f78b4", False),
            ]:
                layer = QgsVectorLayer(f"{crash_file}|layername={table}", f"Level 5 crashes - {label}", "ogr")
                if not layer.isValid():
                    raise RuntimeError(f"Invalid Level 5 crash layer: {table}")
                if table == "junctions_with_crash_counts":
                    symbol = QgsFillSymbol.createSimple({"color": "51,160,44,50", "outline_color": color, "outline_width": "0.4"})
                elif table == "segments_with_crash_counts":
                    symbol = QgsLineSymbol.createSimple({"line_color": color, "line_width": "0.5"})
                else:
                    symbol = QgsMarkerSymbol.createSimple({"name": "circle", "color": color, "outline_color": "white", "outline_width": "0.15", "size": "2.0"})
                layer.setRenderer(QgsSingleSymbolRenderer(symbol))
                project.addMapLayer(layer, False)
                crash_group.addLayer(layer).setItemVisibilityChecked(visible)
                added.append(layer.id())
        project.setTitle(f"{city} street network - Levels 0 to 5 - Important junction approaches - {stamp}")
        if not project.write(str(destination)):
            raise RuntimeError(f"Could not write QGIS project: {destination}")
        # Verify the saved project, including preservation of all previous layers.
        if not project.read(str(destination)):
            raise RuntimeError("Saved QGIS project cannot be reopened")
        assert original_ids <= set(project.mapLayers()), "An original layer was lost"
        assert set(added) <= set(project.mapLayers()), "A Level 5 layer was lost"
        for layer_id in added:
            if not project.mapLayer(layer_id).isValid():
                raise RuntimeError(f"Saved Level 5 layer is invalid: {layer_id}")
        print(f"Saved and reopened QGIS project: {destination}", flush=True)
        print(f"Preserved {len(original_ids)} existing layers; added {len(added)} Level 5 layers.", flush=True)
        project.clear()
        return destination
    finally:
        application.exitQgis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--city", default="Dresden")
    parser.add_argument("--date", help="YYYYMMDD; defaults to today's date in Europe/Berlin")
    arguments = parser.parse_args()
    write_project(arguments.data_root, arguments.city, arguments.date)
