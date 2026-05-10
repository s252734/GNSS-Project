# -*- coding: utf-8 -*-
"""
GNSS pathway visualization for SEPT0642 moving session with outlier filtering.

Inputs:
1. solution_OLS_L1_SEPT0642.csv
2. solution_HATCH_IF_SEPT0642.csv

Outputs:
1. gnss_trajectory_map_SEPT0642_filtered.html
2. trajectory_EN_comparison_SEPT0642_filtered.png
3. correction_magnitude_SEPT0642_filtered.png
4. trajectory_latlon_comparison_SEPT0642_filtered.csv
5. trajectory_outliers_removed_SEPT0642.csv

The map shows:
1. Raw OLS L1 path
2. Corrected HATCH IF path
3. Start and End markers
4. Correction vectors from raw to corrected position
5. Time progression points

Important:
This script removes unrealistic outlier epochs before visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from branca.colormap import linear


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = r"d:\M.Sc. Autonomous Systems - DTU\Spring Semester\30554 GNSS\Lab\lab6"

RAW_CSV = os.path.join(BASE_DIR, "solution_OLS_L1_SEPT0642.csv")
CORR_CSV = os.path.join(BASE_DIR, "solution_HATCH_IF_SEPT0642.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "trajectory_outputs_SEPT0642_filtered")

MAP_HTML = os.path.join(OUTPUT_DIR, "gnss_trajectory_map_SEPT0642_filtered.html")
TRAJ_PLOT_PNG = os.path.join(OUTPUT_DIR, "trajectory_EN_comparison_SEPT0642_filtered.png")
CORR_PLOT_PNG = os.path.join(OUTPUT_DIR, "correction_magnitude_SEPT0642_filtered.png")
LATLON_CSV = os.path.join(OUTPUT_DIR, "trajectory_latlon_comparison_SEPT0642_filtered.csv")
OUTLIERS_CSV = os.path.join(OUTPUT_DIR, "trajectory_outliers_removed_SEPT0642.csv")

POINT_STEP = 20
VECTOR_STEP = 40

SHOW_PLOTS = True

MAP_TILES = "OpenStreetMap"

# Outlier filtering thresholds
MAX_CORRECTION_3D_M = 100.0
MAX_CORRECTED_STEP_M = 20.0
MAX_RAW_STEP_M = 50.0


# =============================================================================
# BASIC FILE FUNCTIONS
# =============================================================================

def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_solution_csv(path):
    """
    Read GNSS solution CSV.

    Expected format:
    index = timestamp
    columns = X, Y, Z, cdt
    """

    df = pd.read_csv(path, index_col=0)

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()

    required_cols = ["X", "Y", "Z"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in file: {path}")

    return df


# =============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# =============================================================================

def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF X, Y, Z to latitude, longitude and height.

    Returns:
    latitude in degrees
    longitude in degrees
    height in metres
    """

    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)

    lat = np.arctan2(z, p * (1.0 - e2))

    for _ in range(10):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N

        lat_new = np.arctan2(z, p * (1.0 - e2 * N / (N + h)))

        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break

        lat = lat_new

    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N

    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg, h


def add_geodetic_columns(df, prefix):
    """
    Add latitude, longitude and height columns.
    """

    df = df.copy()

    lat_list = []
    lon_list = []
    h_list = []

    for _, row in df.iterrows():
        lat, lon, h = ecef_to_geodetic(row["X"], row["Y"], row["Z"])
        lat_list.append(lat)
        lon_list.append(lon)
        h_list.append(h)

    df[f"{prefix}_lat"] = lat_list
    df[f"{prefix}_lon"] = lon_list
    df[f"{prefix}_h"] = h_list

    return df


def ecef_to_enu_matrix(ref_xyz):
    """
    Build rotation matrix from ECEF delta to local ENU.
    """

    lat_deg, lon_deg, _ = ecef_to_geodetic(ref_xyz[0], ref_xyz[1], ref_xyz[2])

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)

    R = np.array([
        [-slon,          clon,          0.0],
        [-slat * clon,  -slat * slon,   clat],
        [ clat * clon,   clat * slon,   slat]
    ])

    return R


def ecef_delta_to_enu(delta_xyz, ref_xyz):
    """
    Convert ECEF difference vector to local ENU.
    """

    R = ecef_to_enu_matrix(ref_xyz)

    if delta_xyz.ndim == 1:
        return R @ delta_xyz

    return (R @ delta_xyz.T).T


# =============================================================================
# DATA PREPARATION
# =============================================================================

def build_time_aligned_dataframe(raw_df, corr_df):
    """
    Align raw and corrected solution dataframes using common timestamps.
    """

    raw = raw_df[["X", "Y", "Z", "raw_lat", "raw_lon", "raw_h"]].copy()
    corr = corr_df[["X", "Y", "Z", "corr_lat", "corr_lon", "corr_h"]].copy()

    raw = raw.rename(columns={
        "X": "raw_X",
        "Y": "raw_Y",
        "Z": "raw_Z"
    })

    corr = corr.rename(columns={
        "X": "corr_X",
        "Y": "corr_Y",
        "Z": "corr_Z"
    })

    df = raw.join(corr, how="inner")
    df = df.sort_index()

    return df


def compute_correction_values(df):
    """
    Compute ENU trajectory and correction vector from raw to corrected.
    """

    df = df.copy()

    ref_xyz = np.array([
        df.iloc[0]["corr_X"],
        df.iloc[0]["corr_Y"],
        df.iloc[0]["corr_Z"]
    ], dtype=float)

    raw_xyz = df[["raw_X", "raw_Y", "raw_Z"]].to_numpy(dtype=float)
    corr_xyz = df[["corr_X", "corr_Y", "corr_Z"]].to_numpy(dtype=float)

    raw_delta = raw_xyz - ref_xyz[None, :]
    corr_delta = corr_xyz - ref_xyz[None, :]

    raw_enu = ecef_delta_to_enu(raw_delta, ref_xyz)
    corr_enu = ecef_delta_to_enu(corr_delta, ref_xyz)

    df["raw_E"] = raw_enu[:, 0]
    df["raw_N"] = raw_enu[:, 1]
    df["raw_U"] = raw_enu[:, 2]

    df["corr_E"] = corr_enu[:, 0]
    df["corr_N"] = corr_enu[:, 1]
    df["corr_U"] = corr_enu[:, 2]

    correction_xyz = corr_xyz - raw_xyz
    correction_enu = ecef_delta_to_enu(correction_xyz, ref_xyz)

    df["dE"] = correction_enu[:, 0]
    df["dN"] = correction_enu[:, 1]
    df["dU"] = correction_enu[:, 2]

    df["horizontal_correction_m"] = np.sqrt(df["dE"]**2 + df["dN"]**2)
    df["vertical_correction_m"] = np.abs(df["dU"])
    df["correction_3d_m"] = np.sqrt(df["dE"]**2 + df["dN"]**2 + df["dU"]**2)

    df["raw_step_m"] = np.sqrt(df["raw_E"].diff()**2 + df["raw_N"].diff()**2)
    df["corr_step_m"] = np.sqrt(df["corr_E"].diff()**2 + df["corr_N"].diff()**2)

    return df


def remove_position_outliers(
    df,
    max_correction_3d_m=MAX_CORRECTION_3D_M,
    max_corrected_step_m=MAX_CORRECTED_STEP_M,
    max_raw_step_m=MAX_RAW_STEP_M
):
    """
    Remove unrealistic GNSS epochs before visualization.

    Filters:
    1. correction_3d_m must be below max_correction_3d_m
    2. corrected step distance must be below max_corrected_step_m
    3. raw step distance must be below max_raw_step_m

    This is only for visualization of the moving trajectory.
    """

    df = df.copy()

    before_count = len(df)

    mask_correction = df["correction_3d_m"] < max_correction_3d_m

    mask_corr_step = (
        df["corr_step_m"].isna()
        | (df["corr_step_m"] < max_corrected_step_m)
    )

    mask_raw_step = (
        df["raw_step_m"].isna()
        | (df["raw_step_m"] < max_raw_step_m)
    )

    final_mask = mask_correction & mask_corr_step & mask_raw_step

    outliers = df[~final_mask].copy()
    filtered = df[final_mask].copy()

    after_count = len(filtered)
    removed_count = before_count - after_count

    print("")
    print("Outlier filtering summary:")
    print(f"Initial epochs: {before_count}")
    print(f"Filtered epochs: {after_count}")
    print(f"Removed epochs: {removed_count}")
    print(f"Max allowed 3D correction: {max_correction_3d_m} m")
    print(f"Max allowed corrected step: {max_corrected_step_m} m")
    print(f"Max allowed raw step: {max_raw_step_m} m")

    if len(filtered) == 0:
        raise ValueError("All epochs were removed by filtering. Relax the thresholds.")

    return filtered, outliers


def print_diagnostics(df, name):
    """
    Print useful diagnostic values before and after filtering.
    """

    print("")
    print(f"Diagnostics for {name}:")
    print(f"Epochs: {len(df)}")

    cols = [
        "horizontal_correction_m",
        "vertical_correction_m",
        "correction_3d_m",
        "raw_step_m",
        "corr_step_m"
    ]

    available_cols = [c for c in cols if c in df.columns]

    print(df[available_cols].describe())

    print("")
    print("Largest 10 correction epochs:")
    print(
        df.sort_values("correction_3d_m", ascending=False)
        [["horizontal_correction_m", "vertical_correction_m", "correction_3d_m", "raw_step_m", "corr_step_m"]]
        .head(10)
    )


# =============================================================================
# STATIC PLOTS
# =============================================================================

def make_static_trajectory_plot(df, out_png):
    """
    Plot raw and corrected trajectories in local East North coordinates.
    """

    plt.figure(figsize=(8, 8))

    plt.plot(
        df["raw_E"],
        df["raw_N"],
        label="Raw OLS L1",
        linewidth=1.5
    )

    plt.plot(
        df["corr_E"],
        df["corr_N"],
        label="Corrected HATCH IF",
        linewidth=2.2
    )

    plt.scatter(
        df["corr_E"].iloc[0],
        df["corr_N"].iloc[0],
        marker="o",
        s=90,
        label="Start"
    )

    plt.scatter(
        df["corr_E"].iloc[-1],
        df["corr_N"].iloc[-1],
        marker="x",
        s=90,
        label="End"
    )

    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("GNSS Receiver Pathway: Raw vs Corrected")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=250)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def make_correction_magnitude_plot(df, out_png):
    """
    Plot correction magnitude over time.
    """

    plt.figure(figsize=(11, 4.5))

    plt.plot(
        df.index,
        df["horizontal_correction_m"],
        label="Horizontal correction"
    )

    plt.plot(
        df.index,
        df["correction_3d_m"],
        label="3D correction"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Correction magnitude [m]")
    plt.title("Real Time Position Correction Magnitude, Filtered")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=250)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def make_step_distance_plot(df, out_png):
    """
    Plot raw and corrected step distance.
    """

    plt.figure(figsize=(11, 4.5))

    plt.plot(df.index, df["raw_step_m"], label="Raw step distance")
    plt.plot(df.index, df["corr_step_m"], label="Corrected step distance")

    plt.xlabel("Epoch")
    plt.ylabel("Step distance [m]")
    plt.title("Step Distance Between Consecutive Epochs, Filtered")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_png, dpi=250)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# =============================================================================
# INTERACTIVE MAP
# =============================================================================

def make_colormap(n):
    """
    Create robust colormap compatible with different branca versions.
    """

    try:
        cmap = linear.viridis.scale(0, max(n - 1, 1))
    except AttributeError:
        cmap = linear.YlOrRd_09.scale(0, max(n - 1, 1))

    cmap.caption = "Time progression"
    return cmap


def create_interactive_map(df, out_html):
    """
    Create interactive map with raw path, corrected path and correction vectors.
    """

    center_lat = df["corr_lat"].mean()
    center_lon = df["corr_lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        tiles=MAP_TILES
    )

    raw_coords = list(zip(df["raw_lat"], df["raw_lon"]))
    corr_coords = list(zip(df["corr_lat"], df["corr_lon"]))

    raw_group = folium.FeatureGroup(name="Raw OLS L1 path")
    corr_group = folium.FeatureGroup(name="Corrected HATCH IF path")
    vector_group = folium.FeatureGroup(name="Correction vectors")
    point_group = folium.FeatureGroup(name="Time progression points")

    folium.PolyLine(
        raw_coords,
        color="blue",
        weight=3,
        opacity=0.75,
        tooltip="Raw OLS L1 path"
    ).add_to(raw_group)

    folium.PolyLine(
        corr_coords,
        color="red",
        weight=4,
        opacity=0.9,
        tooltip="Corrected HATCH IF path"
    ).add_to(corr_group)

    n = len(df)
    cmap = make_colormap(n)

    # Time progression points on corrected path
    for i, (ts, row) in enumerate(df.iloc[::POINT_STEP].iterrows()):
        color = cmap(min(i * POINT_STEP, n - 1))

        popup_text = (
            f"Time: {ts}<br>"
            f"Corrected lat: {row['corr_lat']:.8f}<br>"
            f"Corrected lon: {row['corr_lon']:.8f}<br>"
            f"Horizontal correction: {row['horizontal_correction_m']:.2f} m<br>"
            f"3D correction: {row['correction_3d_m']:.2f} m"
        )

        folium.CircleMarker(
            location=[row["corr_lat"], row["corr_lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup_text
        ).add_to(point_group)

    # Correction vectors from raw to corrected
    for ts, row in df.iloc[::VECTOR_STEP].iterrows():

        raw_point = [row["raw_lat"], row["raw_lon"]]
        corr_point = [row["corr_lat"], row["corr_lon"]]

        popup_text = (
            f"Time: {ts}<br>"
            f"Horizontal correction: {row['horizontal_correction_m']:.2f} m<br>"
            f"Vertical correction: {row['vertical_correction_m']:.2f} m<br>"
            f"3D correction: {row['correction_3d_m']:.2f} m"
        )

        folium.PolyLine(
            locations=[raw_point, corr_point],
            color="green",
            weight=2,
            opacity=0.8,
            popup=popup_text
        ).add_to(vector_group)

        folium.CircleMarker(
            location=raw_point,
            radius=2,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.8
        ).add_to(vector_group)

        folium.CircleMarker(
            location=corr_point,
            radius=2,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.8
        ).add_to(vector_group)

    # Start and End markers
    folium.Marker(
        location=[df["corr_lat"].iloc[0], df["corr_lon"].iloc[0]],
        popup="START",
        tooltip="START",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    folium.Marker(
        location=[df["corr_lat"].iloc[-1], df["corr_lon"].iloc[-1]],
        popup="END",
        tooltip="END",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    raw_group.add_to(m)
    corr_group.add_to(m)
    vector_group.add_to(m)
    point_group.add_to(m)

    cmap.add_to(m)

    folium.LayerControl().add_to(m)

    all_lats = list(df["raw_lat"]) + list(df["corr_lat"])
    all_lons = list(df["raw_lon"]) + list(df["corr_lon"])

    m.fit_bounds([
        [min(all_lats), min(all_lons)],
        [max(all_lats), max(all_lons)]
    ])

    m.save(out_html)


# =============================================================================
# MAIN
# =============================================================================

def main():

    ensure_output_dir(OUTPUT_DIR)

    print("Reading SEPT0642 solution files...")

    raw_df = read_solution_csv(RAW_CSV)
    corr_df = read_solution_csv(CORR_CSV)

    print(f"Raw epochs: {len(raw_df)}")
    print(f"Corrected epochs: {len(corr_df)}")

    print("Converting ECEF to latitude and longitude...")

    raw_df = add_geodetic_columns(raw_df, prefix="raw")
    corr_df = add_geodetic_columns(corr_df, prefix="corr")

    print("Aligning raw and corrected solutions by time...")

    df = build_time_aligned_dataframe(raw_df, corr_df)

    if len(df) == 0:
        raise ValueError("No common timestamps found between raw and corrected CSV files.")

    print(f"Common epochs before filtering: {len(df)}")

    print("Computing correction vectors and ENU trajectory...")

    df = compute_correction_values(df)

    print_diagnostics(df, "before filtering")

    df_filtered, df_outliers = remove_position_outliers(
        df,
        max_correction_3d_m=MAX_CORRECTION_3D_M,
        max_corrected_step_m=MAX_CORRECTED_STEP_M,
        max_raw_step_m=MAX_RAW_STEP_M
    )

    print_diagnostics(df_filtered, "after filtering")

    print("Saving filtered trajectory CSV...")
    df_filtered.to_csv(LATLON_CSV)

    print("Saving removed outlier epochs CSV...")
    df_outliers.to_csv(OUTLIERS_CSV)

    print("Creating static trajectory plot...")
    make_static_trajectory_plot(df_filtered, TRAJ_PLOT_PNG)

    print("Creating correction magnitude plot...")
    make_correction_magnitude_plot(df_filtered, CORR_PLOT_PNG)

    step_plot_path = os.path.join(OUTPUT_DIR, "step_distance_SEPT0642_filtered.png")
    print("Creating step distance plot...")
    make_step_distance_plot(df_filtered, step_plot_path)

    print("Creating interactive map...")
    create_interactive_map(df_filtered, MAP_HTML)

    print("")
    print("Done.")
    print(f"Saved interactive map: {MAP_HTML}")
    print(f"Saved EN trajectory plot: {TRAJ_PLOT_PNG}")
    print(f"Saved correction magnitude plot: {CORR_PLOT_PNG}")
    print(f"Saved step distance plot: {step_plot_path}")
    print(f"Saved filtered combined CSV: {LATLON_CSV}")
    print(f"Saved removed outliers CSV: {OUTLIERS_CSV}")


if __name__ == "__main__":
    main()