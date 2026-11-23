from datetime import datetime
import math
import os
from pathlib import PurePosixPath, Path
import re
from urllib.parse import urlparse
import warnings
from shapely import wkt
from shapely.geometry import Polygon
from rasterio.warp import transform_bounds
import earthaccess
import h5py

NISAR_TS_RE = re.compile(r"_(\d{8}T\d{6})_")

def nisar_start_time_from_url(s3_url: str) -> datetime:
    path = urlparse(s3_url).path
    name = PurePosixPath(path).name
    
    m = NISAR_TS_RE.search(name)
    if not m:
        raise ValueError(f"No NISAR timestamp found in: {s3_url}")
    
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")


def gcov_spacing(result, group="/science/LSAR/GCOV/grids/frequencyA"):
    """Grid spacing (m) of one GCOV granule; reads one scalar, no pixel data."""
    f = earthaccess.open([result])[0]
    try:
        with h5py.File(f, "r") as h5:
            return abs(float(h5[f"{group}/xCoordinateSpacing"][()]))
    finally:
        f.close()


def _n_touched(lo, hi, spacing, edge_offset):
    """Pixels whose extent touches [lo, hi]; matches rio.clip_box rounding."""
    return math.ceil((hi - edge_offset) / spacing) - math.floor((lo - edge_offset) / spacing)


def _footprint(result):
    """Granule footprint polygon (lon/lat) from an earthaccess search result."""
    pts = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"][
        "GPolygons"][0]["Boundary"]["Points"]
    return Polygon([(p["Longitude"], p["Latitude"]) for p in pts])


def estimate_dataset_size(aoi_wkt, results, spacing=10.0, variables=("HHHH", "HVHV"),
                          itemsize=4, epsg=None, x_edge_offset=0.0, y_edge_offset=0.0):
    """Estimate the shape/size of the clipped, concatenated GCOV time series."""
    aoi = wkt.loads(aoi_wkt)
    if epsg is None:  # UTM zone of the AOI centroid
        lon, lat = aoi.centroid.x, aoi.centroid.y
        epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    dst = f"EPSG:{epsg}"
    ax0, ay0, ax1, ay1 = transform_bounds("EPSG:4326", dst, *aoi.bounds)

    ux0 = uy0 = math.inf
    ux1 = uy1 = -math.inf
    n = 0
    for r in results:
        fx0, fy0, fx1, fy1 = transform_bounds("EPSG:4326", dst, *_footprint(r).bounds)
        x0, y0, x1, y1 = max(ax0, fx0), max(ay0, fy0), min(ax1, fx1), min(ay1, fy1)
        if x0 >= x1 or y0 >= y1:
            continue  # product doesn't overlap the AOI
        n += 1
        ux0, uy0, ux1, uy1 = min(ux0, x0), min(uy0, y0), max(ux1, x1), max(uy1, y1)

    if n == 0:
        return {"epsg": epsg, "shape": (0, 0, 0), "nbytes": 0}
    nx = _n_touched(ux0, ux1, spacing, x_edge_offset)
    ny = _n_touched(uy0, uy1, spacing, y_edge_offset)
    return {"epsg": epsg, "shape": (n, ny, nx),
            "nbytes": n * ny * nx * itemsize * len(variables)}

def get_available_memory():
    """
    RAM a notebook can still use before the pod is OOM-killed.

    Reads the container's cgroup (v2, then v1), so it reports the pod's limit,
    not the node's total like psutil/top do. Usage is the "working set"
    (usage minus reclaimable page cache), the number Kubernetes kills on.
    Returns a dict of bytes: limit, used, available.
    """
    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return None

    def stat(p, key):
        for line in (read(p) or "").splitlines():
            k, _, v = line.partition(" ")
            if k == key:
                return int(v)
        return 0

    limit = used = None

    # cgroup v2
    v = read("/sys/fs/cgroup/memory.max")
    if v and v != "max":
        limit = int(v)
    v = read("/sys/fs/cgroup/memory.current")
    if v:
        used = int(v) - stat("/sys/fs/cgroup/memory.stat", "inactive_file")

    # cgroup v1
    if limit is None:
        v = read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if v and int(v) < 2**60:
            limit = int(v)
    if used is None:
        v = read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if v:
            used = int(v) - stat("/sys/fs/cgroup/memory/memory.stat", "total_inactive_file")

    # fallbacks: JupyterHub KubeSpawner env var, then host RAM
    if limit is None and os.environ.get("MEM_LIMIT", "").isdigit():
        limit = int(os.environ["MEM_LIMIT"])
    if limit is None:
        limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if used is None:
        import psutil
        used = psutil.Process().memory_info().rss

    used = max(used, 0)
    return {"limit": limit, "used": used, "available": max(limit - used, 0)}

def assess_notebook_resources(dataset_size_est, notebook_peak_factor,
                              baseline=0.55e9, node_reserve=0.20):
    """
    Raise MemoryError if running the notebook on this dataset is likely to
    exceed the server's memory. Call it right before loading the data.
    """
    mem = get_available_memory()
    # memory already held by Jupyter + this kernel (at least the idle baseline)
    in_use = max(mem["used"], baseline)
    budget = (1 - node_reserve) * mem["limit"] - in_use
    need = notebook_peak_factor * dataset_size_est["nbytes"]

    print(f"Estimated dataset: {dataset_size_est['shape']}, "
          f"{dataset_size_est['nbytes']/1e9:.2f} GB")
    print(f"Memory already in use: {in_use/1e9:.1f} GB | "
          f"Remaining available memory: {budget/1e9:.1f} GB | "
          f"Estimated peak notebook memory usage: {need/1e9:.1f} GB")

    if need > budget:
        max_gb = max(budget, 0) / notebook_peak_factor / 1e9
        warnings.warn(
            "Dataset too large given your available memory resources."
            "\nIf you are on a JupyterHub, continuing to run this notebook may cause your kernel or server to crash."
            f"\nMaximum suggested dataset size: ~{max_gb:.2f} GB"
            f"\nEstimated dataset size: {dataset_size_est["nbytes"]/1e9:.2f} GB"
            "\nConsider filtering by track and frame, shrinking your AOI, and/or using fewer dates.",
            NotebookMemoryWarning,
            stacklevel=2,
        )

class NotebookMemoryWarning(UserWarning):
    """The dataset may be too large for this server's memory."""

# make sure it shows every time, not just the first time per line
warnings.simplefilter("always", NotebookMemoryWarning)