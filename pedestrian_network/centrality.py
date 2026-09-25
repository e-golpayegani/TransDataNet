"""
centrality.py

Purpose
-------
Create a separate copy of the processed street network for sDNA
centrality analysis.

The city is read automatically from config.yml using config_data.

The script:

1. Reads the city from config.yml.
2. Finds the latest street_net_<city>_v*.gpkg in the city's draft folder.
3. Loads that street network.
4. Performs basic diagnostic checks.
5. Adds a stable "centrality_id".
6. Records which street-network file was used as the source.
7. Saves a separate centrality street network in the draft folder.

The output can then be used directly as the input network in
sDNA Integral in QGIS.

IMPORTANT
---------
This script does NOT:

- simplify the network
- clean the network
- snap lines
- split lines
- merge lines
- remove lines
- change topology
- change geometry
- calculate centrality

Closeness/farness and betweenness are calculated manually using
sDNA Integral in QGIS.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import re

import geopandas as gpd

from src.utils.config_loader import config_data



# ============================================================
# SETTINGS
# ============================================================

# City is read automatically from config.yml.
#
# This is the same approach used in main.py.
#
# Example:
#
# city_name: Leipzig
#
# -> CITY = "Leipzig"

CITY = config_data["city_name"]


# Version of the centrality-specific street network.
#
# This version is independent from the version of the source
# street network.

CENTRALITY_VERSION = "1.0"


# ============================================================
# PATHS
# ============================================================

# centrality.py is located in:
#
# pedestrian_network/
#
# Therefore the project directory is the directory containing
# this script.

PROJECT_DIR = Path(__file__).resolve().parent


# Existing project output structure:
#
# pedestrian_network/
# └── src/
#     └── data/
#         └── output/
#             └── Leipzig/
#                 └── draft/

OUTPUT_ROOT = (
    PROJECT_DIR
    / "src"
    / "data"
    / "output"
)


CITY_DIR = (
    OUTPUT_ROOT
    / CITY
)


DRAFT_DIR = (
    CITY_DIR
    / "draft"
)


# Check that the city output directory exists.

if not CITY_DIR.exists():

    raise FileNotFoundError(
        "\n"
        f"City output directory does not exist:\n"
        f"{CITY_DIR}\n\n"
        f"Check city_name in config.yml.\n"
        f"Current city: {CITY}"
    )


# Check that draft directory exists.

if not DRAFT_DIR.exists():

    raise FileNotFoundError(
        "\n"
        f"Draft directory does not exist:\n"
        f"{DRAFT_DIR}"
    )


# Output of THIS script.

CENTRALITY_NETWORK_PATH = (
    DRAFT_DIR
    / (
        f"centrality_street_net_"
        f"{CITY}_"
        f"v{CENTRALITY_VERSION}.gpkg"
    )
)


# ============================================================
# VERSION HANDLING
# ============================================================

def get_version(path):
    """
    Extract the version number from a versioned GeoPackage.

    Examples
    --------

    street_net_Leipzig_v1.0.gpkg
        -> (1, 0)

    street_net_Leipzig_v1.2.gpkg
        -> (1, 2)

    street_net_Leipzig_v2.3.gpkg
        -> (2, 3)


    Parameters
    ----------
    path : pathlib.Path
        Path to the GeoPackage.


    Returns
    -------
    tuple
        (major_version, minor_version)

        Returns (-1, -1) if no valid version number
        can be found.
    """

    match = re.search(
        r"_v(\d+)\.(\d+)\.gpkg$",
        path.name,
        flags=re.IGNORECASE,
    )

    if match is None:

        return (-1, -1)

    major = int(
        match.group(1)
    )

    minor = int(
        match.group(2)
    )

    return (
        major,
        minor,
    )


# ============================================================
# FIND LATEST STREET NETWORK
# ============================================================

def find_latest_street_network():
    """
    Find the latest processed street network for the city.

    Search location
    ---------------

    src/data/output/<CITY>/draft/


    Expected naming
    ---------------

    street_net_<CITY>_v*.gpkg


    Example
    -------

    If draft contains:

        street_net_Leipzig_v1.0.gpkg
        street_net_Leipzig_v1.1.gpkg
        street_net_Leipzig_v1.2.gpkg

    this function returns:

        street_net_Leipzig_v1.2.gpkg
    """

    pattern = (
        f"street_net_{CITY}_v*.gpkg"
    )


    candidates = list(
        DRAFT_DIR.glob(
            pattern
        )
    )


    if not candidates:

        raise FileNotFoundError(
            "\n"
            "No processed street network found.\n\n"
            f"City:\n"
            f"  {CITY}\n\n"
            f"Folder searched:\n"
            f"  {DRAFT_DIR}\n\n"
            f"Expected filename pattern:\n"
            f"  street_net_{CITY}_v*.gpkg"
        )


    latest = max(
        candidates,
        key=get_version,
    )


    return latest


# ============================================================
# NETWORK CHECK
# ============================================================

def check_network(gdf):
    """
    Perform basic diagnostic checks before the network
    is sent to sDNA.

    IMPORTANT
    ---------

    This function only REPORTS problems.

    It does not:

    - repair geometry
    - simplify geometry
    - snap geometry
    - remove geometry
    - modify topology
    """


    print(
        "\n"
        "============================================================"
    )

    print(
        "NETWORK CHECK"
    )

    print(
        "============================================================"
    )


    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if gdf.empty:

        raise ValueError(
            "The street network is empty."
        )


    print(
        f"\nNumber of street features: "
        f"{len(gdf):,}"
    )


    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    if gdf.crs is None:

        raise ValueError(
            "\nThe street network has no CRS."
        )


    print(
        f"\nCRS:\n"
        f"  {gdf.crs}"
    )


    if gdf.crs.is_geographic:

        print(
            "\nWARNING:"
            "\nThe network uses a geographic CRS."
            "\nFor metric-radius centrality analysis, "
            "a projected CRS in metres is recommended."
        )

    else:

        print(
            "\nCRS type:"
            "\n  Projected"
        )


    # --------------------------------------------------------
    # Geometry column
    # --------------------------------------------------------

    if "geometry" not in gdf.columns:

        raise ValueError(
            "No geometry column found."
        )


    # --------------------------------------------------------
    # Geometry types
    # --------------------------------------------------------

    geometry_types = (
        gdf.geometry
        .geom_type
        .value_counts()
    )


    print(
        "\nGeometry types:"
    )


    for geometry_type, count in geometry_types.items():

        print(
            f"  {geometry_type}: "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Missing geometries
    # --------------------------------------------------------

    missing_geometry = (
        gdf.geometry.isna()
    )


    number_missing = int(
        missing_geometry.sum()
    )


    print(
        f"\nMissing geometries: "
        f"{number_missing:,}"
    )


    # --------------------------------------------------------
    # Empty geometries
    # --------------------------------------------------------

    empty_geometry = (
        gdf.geometry.is_empty
    )


    number_empty = int(
        empty_geometry.sum()
    )


    print(
        f"Empty geometries: "
        f"{number_empty:,}"
    )


    # --------------------------------------------------------
    # Invalid geometries
    # --------------------------------------------------------

    invalid_geometry = (
        ~gdf.geometry.is_valid
    )


    number_invalid = int(
        invalid_geometry.sum()
    )


    print(
        f"Invalid geometries: "
        f"{number_invalid:,}"
    )


    if number_invalid > 0:

        print(
            "\nWARNING:"
            "\nInvalid geometries were detected."
            "\nThey are NOT being repaired automatically."
        )


    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    if (
        number_missing > 0
        or number_empty > 0
    ):

        print(
            "\nWARNING:"
            "\nMissing or empty geometries were detected."
            "\nNothing has been removed automatically."
        )


# ============================================================
# CENTRALITY ID
# ============================================================

def add_centrality_id(gdf):
    """
    Create a stable identifier for the centrality network.

    This identifier allows the sDNA result to be linked
    back to the street network.

    If centrality_id already exists, it is checked and
    preserved.
    """


    gdf = gdf.copy()


    # --------------------------------------------------------
    # Existing ID
    # --------------------------------------------------------

    if "centrality_id" in gdf.columns:

        print(
            "\nExisting 'centrality_id' found."
        )


        # Missing IDs

        number_missing_ids = int(
            gdf[
                "centrality_id"
            ]
            .isna()
            .sum()
        )


        if number_missing_ids > 0:

            raise ValueError(
                "\n"
                "Existing centrality_id contains "
                f"{number_missing_ids:,} missing values."
            )


        # Duplicate IDs

        number_duplicate_ids = int(
            gdf[
                "centrality_id"
            ]
            .duplicated()
            .sum()
        )


        if number_duplicate_ids > 0:

            raise ValueError(
                "\n"
                "Existing centrality_id contains "
                f"{number_duplicate_ids:,} duplicates."
            )


        print(
            "Existing centrality_id is valid."
        )


        return gdf


    # --------------------------------------------------------
    # New ID
    # --------------------------------------------------------

    gdf = gdf.reset_index(
        drop=True
    )


    gdf[
        "centrality_id"
    ] = (
        gdf.index + 1
    )


    print(
        "\nCreated new 'centrality_id'."
    )


    return gdf


# ============================================================
# CREATE CENTRALITY STREET NETWORK
# ============================================================

def create_centrality_network():
    """
    Main workflow.
    """


    print(
        "\n"
        "============================================================"
    )

    print(
        "CENTRALITY STREET NETWORK"
    )

    print(
        "============================================================"
    )


    print(
        f"\nCity from config.yml:"
        f"\n  {CITY}"
    )


    # --------------------------------------------------------
    # Find latest street network
    # --------------------------------------------------------

    source_path = (
        find_latest_street_network()
    )


    source_version = (
        get_version(
            source_path
        )
    )


    print(
        "\nLatest processed street network:"
    )


    print(
        f"  {source_path.name}"
    )


    print(
        f"\nDetected source version:"
        f"\n  v{source_version[0]}."
        f"{source_version[1]}"
    )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading network..."
    )


    streets = gpd.read_file(
        source_path
    )


    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    check_network(
        streets
    )


    # --------------------------------------------------------
    # Stable ID
    # --------------------------------------------------------

    streets = (
        add_centrality_id(
            streets
        )
    )


    # --------------------------------------------------------
    # Record provenance
    # --------------------------------------------------------

    streets[
        "centrality_source"
    ] = (
        source_path.name
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No geometry operation occurs between reading the source
    # and writing the output.
    #
    # Therefore the geometries in the centrality network are
    # copies of those in the main.py street network.
    # --------------------------------------------------------


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving centrality street network..."
    )


    streets.to_file(
        CENTRALITY_NETWORK_PATH,
        layer="centrality_street_network",
        driver="GPKG",
    )


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "DONE"
    )

    print(
        "============================================================"
    )


    print(
        "\nSource network:"
    )

    print(
        f"  {source_path}"
    )


    print(
        "\nCentrality network:"
    )

    print(
        f"  {CENTRALITY_NETWORK_PATH}"
    )


    print(
        f"\nNumber of street features:"
        f"\n  {len(streets):,}"
    )


    print(
        "\nGeometry/topology modifications:"
        "\n  NONE"
    )


    print(
        "\nThe source street network was NOT modified."
    )


    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        "NEXT STEP: sDNA"
    )

    print(
        "------------------------------------------------------------"
    )


    print(
        "\nOpen this GeoPackage in QGIS:"
    )

    print(
        f"\n  {CENTRALITY_NETWORK_PATH}"
    )


    print(
        "\nSelect layer:"
    )

    print(
        "\n  centrality_street_network"
    )


    print(
        "\nUse this layer as the input to:"
    )

    print(
        "\n  sDNA Integral"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "\n  Preserve the 'centrality_id' field "
        "in the sDNA output."
    )


    print(
        "\n"
        "============================================================"
    )


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    create_centrality_network()