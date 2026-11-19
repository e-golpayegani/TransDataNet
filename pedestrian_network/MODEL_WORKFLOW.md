# One network for all pedestrian-volume models

Each city and frozen network version has one modelling GeoPackage:
`model/pedestrian_model_network_<city>_v1.0.gpkg`, layer
`pedestrian_model_network`. One feature represents one street segment.

- `segment_id`: the single modelling ID, assigned once and saved. It is unique
  within a city/network version. Use city + version + segment_id across datasets.
  Existing centrality_id values are renamed without renumbering; street_id is
  removed. If only street_id exists, its values are retained as segment_id.
- `network_source` and `network_version`: identify the attributed source snapshot.
- Original POI counts, length, other collected data, imported centrality measures
  remain on that same feature. Final model predictions use separate layers
  with identical segment IDs and geometries.
- IDs are reused on every run. Changed segmentation requires a new network version;
  it does not establish correspondence to a previous version automatically.
- Do not delete and recreate the model network to refresh attributes.

## Workflow

Run commands from the pedestrian_network folder, with the city set in config.yml.

1. Run the retrieval/processing stages and main.py to collect POIs and other
   attributes. main.py writes the attributed street network in the city output
   folder; draft networks are intermediate and lack the POI attributes.
2. Run `python centrality.py`. It copies the latest attributed street network
   into the model network and exports centrality_input for QGIS. An existing
   matching legacy input is preserved and its ID values adopted as segment_id. Use
   `--source "path/to/street_net.gpkg"` to choose a source explicitly at creation.
   Repeated runs reuse the model network, not a newly discovered source.
3. Run sDNA Integral on that exact input. Save its results permanently, for example
   as `model/centrality_output_<city>_v1.0.gpkg`, layer `centrality_output`.
   This is a manual QGIS result, not another canonical network.
4. Run `python pedestrian_volume.py`. It automatically reads the sDNA export
   at the suggested path, restores omitted segment_id values, attaches centrality
   to the shared attributed network, and calculates every configured model.
   For a different export location or layer, use:
   `python pedestrian_volume.py --sdna-results "path/to/results.gpkg" --layer your_layer`
   No second run of centrality.py is needed.
   The import validates retained IDs against geometry; without IDs it requires
   exact normalized XY geometry matching. Missing/ambiguous matches stop the run.
   If an earlier import already saved the required fields and no default export
   exists, those saved fields can be reused.
5. Results are written to `model/pedestrian_volume_<city>_v1.0.gpkg`, with one
   `model_<name>` layer per selected MODELS entry, plus `model_comparison`.
   The current entries create `model_closeness` and `model_poi`. Adding models
   named `poi_retail` and `poi_hotels`, for example, also creates layers
   `model_poi_retail` and `model_poi_hotels`; supply their own fitted coefficients.
   Every layer retains the same segment_id and street geometries. The shared
   network stores collected predictors; predictions are in the final layers.
   Before sDNA is available, `python pedestrian_volume.py --models poi` runs
   only the POI model without requiring or automatically importing sDNA results.
   No street is removed for missing predictors. Rerun pedestrian_volume.py
   whenever sDNA data or model settings change; exported layers are snapshots.
6. Open the model_comparison layer in QGIS to compare `PV_closeness` and `PV_poi` on the
   same street. All `PV_<name>` predictions are rounded whole numbers. There
   are no duplicate unrounded, linear-predictor or `_count` output fields.
   Integer differences are generated for every pair of selected models and
   remain missing when either model lacks a valid prediction. `comparison_valid`
   indicates that all selected models have a prediction for that segment. Predictive accuracy requires
   observed pedestrian counts, not just agreement between models.

Each MODELS entry in pedestrian_volume.py represents a separate fitted model
with its own intercept, coefficients and transformations. Existing values are
carried over from the previous script: closeness uses 6.263 + 0.904*NQPDA500;
POI uses 6.923 + 0.008*retail/services density + 0.095*hotel density, where density
is count/(length_km*10). Confirm these match the fitted models before interpreting
predictions. Additional POI models require their own fitted coefficients; do not
reuse a combined-model intercept after arbitrarily removing predictors.

Use `--version 2.0` consistently on both scripts for a new frozen network version.
Keep the original model network, matching sDNA input and raw results.

## Identity through sDNA

The saved reference network is authoritative. sDNA may omit custom fields from
its raw output; the script cannot force the external tool to retain them. The
import step restores segment_id by complete, unambiguous geometry matching
before attaching attributes. It never assigns fresh IDs to calculated results
or uses sDNA ID/row order as the modelling key. If a retained ID disagrees with
geometry, the import stops. Do not rename sDNA ID to segment_id.

Run centrality.py once on an existing city to migrate the shared network and
prepared input to the single field. Original numeric centrality_id values and
geometries are preserved, so already calculated sDNA results remain usable.
Rerun pedestrian_volume.py to regenerate the final model layers with segment_id.
Raw sDNA exports and previously generated prediction files are not migrated
until imported/regenerated. Keep the saved reference network throughout.

## Scripts

Only centrality.py and pedestrian_volume.py are needed for the modelling steps.
Segment identity, geometry validation and sDNA import functions live inside
pedestrian_volume.py. centrality.py imports its preparation helpers without
running model calculations. There is no separate model_network.py module.

## Adding more fitted POI models

Add each fitted specification as a uniquely named entry in MODELS in
pedestrian_volume.py, using the same structure as poi. Each entry needs its own
intercept and variables (source_col, coefficient, transform). Omit requires_sdna
or set it to false for POI-only models. Do not change existing model coefficients
to invent an additional model. The default command runs all entries; --models
can select their names. The results file contains exactly the selected model
layers plus model_comparison, so a rerun does not retain stale layers.
