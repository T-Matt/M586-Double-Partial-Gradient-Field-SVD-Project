import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from dask.diagnostics import ProgressBar
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── GODAS (ocean) ─────────────────────────────────────────────────────────────
GODAS_VARIABLES = ["salt", "pottmp", "ucur", "vcur"]
GODAS_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/godas"
GODAS_START_YEAR = 1980

# ── NCEP/DOE Reanalysis II (atmosphere) ───────────────────────────────────────
# Monthly-mean files are single files spanning the full record (1979–present).
# Note: some surface variables start in 1948 — these are trimmed to match GODAS.
R2_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/Monthlies"

R2_SURFACE_VARIABLES = [
    "air.sig995",    # near-surface air temperature
    "pottmp.sig995", # near-surface potential temperature
    "rhum.sig995",   # near-surface relative humidity
    "mslp",          # mean sea-level pressure
    "pres.sfc",      # surface pressure
    "pr_wtr.eatm",   # precipitable water (entire atmosphere)
]

R2_PRESSURE_VARIABLES = [
    "uwnd",  # zonal wind (all pressure levels)
    "vwnd",  # meridional wind (all pressure levels)
    "air",   # air temperature (all pressure levels)
    "hgt",   # geopotential height
    "rhum",  # relative humidity
]

# ── Request settings ───────────────────────────────────────────────────────────
CHUNK_SIZE      = 1024 * 1024  # 1 MB chunks for streaming writes
REQUEST_TIMEOUT = 120           # seconds before a stalled request is aborted
SLEEP_BETWEEN   = 1.0           # seconds to pause between individual file downloads
MAX_RETRIES     = 5             # retry attempts on server errors / transient failures
BACKOFF_FACTOR  = 2             # exponential backoff: waits 2, 4, 8, 16 … seconds

_REDOWNLOAD_HINT = (
    "\n  To redownload from scratch, delete the relevant folder and rerun download_data():\n"
    "    GODAS  -> delete the 'GODAS_data/' folder\n"
    "    R2     -> delete the 'R2_data/' folder\n"
    "  Then call:  from Data_DL.NOAA_Downloader.NOAA_data_download import download_data; download_data()\n"
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Return a Session with automatic retry/backoff and a polite User-Agent."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "xslice-data-downloader/1.0 (research)"})
    return session


def _download_file(session: requests.Session, url: str, dest: Path):
    """Stream a single file from *url* to *dest*, raising on HTTP errors."""
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.unlink(missing_ok=True)
    r = session.get(url, allow_redirects=True, stream=True, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
    tmp.replace(dest)


def _detect_latest_godas_year(session: requests.Session) -> int:
    """Probe the PSL server to find the most recent year with a complete GODAS file.

    Walks backwards from the current year using lightweight HEAD requests
    (no data downloaded) until it finds an existing annual file for 'salt'.
    Falls back to two years before the current year if probing fails.
    """
    current_year = datetime.now().year
    print("Detecting latest available GODAS year from PSL server...")
    for year in range(current_year, GODAS_START_YEAR, -1):
        url = f"{GODAS_BASE_URL}/salt.{year}.nc"
        try:
            r = session.head(url, timeout=15, allow_redirects=True)
            if r.status_code == 200:
                print(f"  Latest available GODAS year: {year}")
                return year
        except requests.RequestException:
            continue
    fallback = current_year - 2
    print(f"  Could not probe server — falling back to {fallback}")
    return fallback


# ── GODAS download ─────────────────────────────────────────────────────────────

def download_godas(folder: Path, latest_year: int, session: requests.Session):
    folder.mkdir(parents=True, exist_ok=True) # if the folder doesn't exist, create it (and any necessary parents)
    combined_files = [folder / f"godas_{var}.nc" for var in GODAS_VARIABLES]

    if all(cf.exists() for cf in combined_files):
        print(
            f"GODAS: combined files already exist, skipping download.\n"
            f"  Coverage: {GODAS_START_YEAR}–{latest_year}" + _REDOWNLOAD_HINT
        )
        create_godas_climatologies(folder)
        return

    years = np.arange(GODAS_START_YEAR, latest_year + 1)

    for year in years:
        for var in GODAS_VARIABLES:
            dest = folder / f"{var}.{year}.nc"
            if not dest.exists():
                url = f"{GODAS_BASE_URL}/{var}.{year}.nc"
                print(f"  Downloading {dest.name}...")
                try:
                    _download_file(session, url, dest)
                except Exception as e:
                    print(f"  ERROR downloading {dest.name}: {e}" + _REDOWNLOAD_HINT)
                time.sleep(SLEEP_BETWEEN)
            else:
                print(f"  {dest.name} already exists, skipping.")

    for var in GODAS_VARIABLES:
        print(f"  Combining {var}.*.nc -> godas_{var}.nc ...")
        yearly_files = sorted(folder.glob(f"{var}.*.nc"))
        ds = xr.open_mfdataset(yearly_files, combine="nested", concat_dim="time", engine="netcdf4", chunks={"time": 12, "level": 1})
        out_nc = folder / f"godas_{var}.nc"
        tmp_nc = out_nc.with_name(out_nc.name + ".tmp")
        tmp_nc.unlink(missing_ok=True)
        with ProgressBar():
            ds.to_netcdf(tmp_nc)
        ds.close()
        tmp_nc.replace(out_nc)
        for f in yearly_files:
            f.unlink()
        print(f"  Deleted {len(yearly_files)} yearly files for {var}.")

    print(f"GODAS download complete. Files in: {folder.resolve()}")
    create_godas_climatologies(folder)


def create_godas_climatologies(folder: Path):
    for var in GODAS_VARIABLES:
        combined_file = folder / f"godas_{var}.nc"
        clim_file     = folder / f"godas_{var}_clim.nc"

        if not combined_file.exists():
            print(f"  Cannot create GODAS climatology for {var} — {combined_file.name} missing!" + _REDOWNLOAD_HINT)
            continue

        if clim_file.exists():
            print(f"  GODAS climatology {clim_file.name} already exists, skipping.")
            continue

        print(f"  Creating GODAS climatology: {var} -> {clim_file.name}")
        ds = xr.open_dataset(combined_file, chunks={"time": 12, "level": 1})

        if not isinstance(ds.time.values[0], np.datetime64):
            ds["time"] = ds.indexes["time"].to_datetimeindex()

        ds_clim = ds.groupby("time.month").mean(dim="time", keep_attrs=True)
        tmp_clim = clim_file.with_name(clim_file.name + ".tmp")
        tmp_clim.unlink(missing_ok=True)
        with ProgressBar():
            ds_clim.to_netcdf(tmp_clim)
        ds.close()
        tmp_clim.replace(clim_file)
        print(f"  Saved: {clim_file.name}")

    print(f"GODAS climatologies in: {folder.resolve()}")


# ── Reanalysis II download ─────────────────────────────────────────────────────

def download_r2(folder: Path, session: requests.Session):
    """Download NCEP/DOE Reanalysis II monthly-mean files (full record per variable)."""
    folder.mkdir(parents=True, exist_ok=True)

    all_vars = (
        [(v, "surface") for v in R2_SURFACE_VARIABLES]
        + [(v, "pressure") for v in R2_PRESSURE_VARIABLES]
    )

    for var, subdir in all_vars:
        filename = f"r2_{var}.mon.mean.nc"
        dest = folder / filename

        trimmed_dest = folder / "R2_trimmed" / f"r2_{var}_trimmed.nc"
        if dest.exists() or trimmed_dest.exists():
            print(f"  R2: {var} already processed, skipping.")
            continue

        url = f"{R2_BASE_URL}/{subdir}/{var}.mon.mean.nc"
        print(f"  Downloading R2 {var} ({subdir}) -> {filename} ...")
        try:
            _download_file(session, url, dest)
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}" + _REDOWNLOAD_HINT)
        time.sleep(SLEEP_BETWEEN)

    print(f"Reanalysis II download complete. Files in: {folder.resolve()}")
    create_r2_climatologies(folder)


def create_r2_climatologies(folder: Path):
    """Compute monthly climatologies from the full (untrimmed) R2 files."""
    all_vars = R2_SURFACE_VARIABLES + R2_PRESSURE_VARIABLES

    for var in all_vars:
        src_file  = folder / f"r2_{var}.mon.mean.nc"
        clim_file = folder / f"r2_{var}_clim.nc"

        if not src_file.exists():
            print(f"  Cannot create R2 climatology for {var} — {src_file.name} missing!" + _REDOWNLOAD_HINT)
            continue

        if clim_file.exists():
            print(f"  R2 climatology {clim_file.name} already exists, skipping.")
            continue

        print(f"  Creating R2 climatology: {var} -> {clim_file.name}")
        ds = xr.open_dataset(src_file)

        if not isinstance(ds.time.values[0], np.datetime64):
            ds["time"] = ds.indexes["time"].to_datetimeindex()

        ds_clim = ds.groupby("time.month").mean(dim="time", keep_attrs=True)
        ds_clim.to_netcdf(clim_file)
        ds.close()
        print(f"  Saved: {clim_file.name}")

    print(f"R2 climatologies in: {folder.resolve()}")


# ── Temporal alignment ─────────────────────────────────────────────────────────

def align_r2_to_godas(godas_folder: Path, r2_folder: Path):
    """Trim R2 files to match the GODAS time window and save into a dedicated folder.

    The GODAS combined files define the reference window (1980-01 -> latest Dec).
    Trimmed files are written to R2_data/R2_trimmed/ to keep them separate from
    the full-record originals. Climatologies are saved alongside as *_clim.nc.
    """
    godas_folder, r2_folder = Path(godas_folder), Path(r2_folder)
    trimmed_folder = r2_folder / "R2_trimmed"
    trimmed_folder.mkdir(exist_ok=True)

    ref_file = godas_folder / f"godas_{GODAS_VARIABLES[0]}.nc"
    if not ref_file.exists():
        print(f"Cannot align — GODAS reference file missing: {ref_file}" + _REDOWNLOAD_HINT)
        return

    ds_ref = xr.open_dataset(ref_file)
    if not isinstance(ds_ref.time.values[0], np.datetime64):
        ds_ref["time"] = ds_ref.indexes["time"].to_datetimeindex()

    time_start = str(ds_ref.time.values[0])[:7]   # e.g. "1980-01"
    time_end   = str(ds_ref.time.values[-1])[:7]  # e.g. "2024-12"
    ds_ref.close()

    print(f"\nAligning R2 to GODAS window: {time_start} -> {time_end}")
    print(f"Trimmed files will be saved to: {trimmed_folder.resolve()}")

    all_vars = R2_SURFACE_VARIABLES + R2_PRESSURE_VARIABLES

    for var in all_vars:
        src     = r2_folder / f"r2_{var}.mon.mean.nc"
        trimmed = trimmed_folder / f"r2_{var}_trimmed.nc"
        clim    = trimmed_folder / f"r2_{var}_trimmed_clim.nc"

        if not src.exists():
            print(f"  MISSING source {src.name} — skipping trim." + _REDOWNLOAD_HINT)
            continue

        if trimmed.exists():
            print(f"  {trimmed.name} already exists, skipping.")
        else:
            print(f"  Trimming {src.name} -> {trimmed.name} ...")
            ds = xr.open_dataset(src)
            if not isinstance(ds.time.values[0], np.datetime64):
                ds["time"] = ds.indexes["time"].to_datetimeindex()
            ds.sel(time=slice(time_start, time_end)).to_netcdf(trimmed)
            ds.close()

        if clim.exists():
            print(f"  {clim.name} already exists, skipping.")
        else:
            print(f"  Creating trimmed climatology: {var} -> {clim.name} ...")
            ds_t = xr.open_dataset(trimmed)
            if not isinstance(ds_t.time.values[0], np.datetime64):
                ds_t["time"] = ds_t.indexes["time"].to_datetimeindex()
            ds_t.groupby("time.month").mean(dim="time", keep_attrs=True).to_netcdf(clim)
            ds_t.close()
            print(f"  Saved: {clim.name}")

        if trimmed.exists() and clim.exists() and src.exists():
            src.unlink()
            print(f"  Deleted original consolidated file: {src.name}")

    print(f"R2 alignment complete. Trimmed files in: {trimmed_folder.resolve()}")


# ── Entry point ────────────────────────────────────────────────────────────────

def download_data(godas_folder: Path, r2_folder: Path):

    session     = _make_session()
    latest_year = _detect_latest_godas_year(session)

    download_godas(godas_folder, latest_year, session)
    download_r2(r2_folder, session)
    align_r2_to_godas(godas_folder, r2_folder)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(path: Path):
    if not path.exists():
        print(f"  MISSING : {path.name}")
        return
    try:
        ds = xr.open_dataset(path)
        dims = dict(ds.sizes)
        coords = list(ds.coords)
        time_range = (
            f"{ds.time.values[0]} -> {ds.time.values[-1]}"
            if "time" in ds.coords else "no time coord"
        )
        print(f"  OK      : {path.name}")
        print(f"            dims={dims}")
        print(f"            coords={coords}")
        print(f"            time={time_range}")
        ds.close()
    except Exception as e:
        print(f"  ERROR   : {path.name} — {e}" + _REDOWNLOAD_HINT)


def validate_files(godas_folder: Path, r2_folder: Path):
    trimmed_folder = r2_folder / "R2_trimmed"

    print("=" * 60)
    print("GODAS files")
    print("=" * 60)
    for var in GODAS_VARIABLES:
        validate(godas_folder / f"godas_{var}.nc")
        validate(godas_folder / f"godas_{var}_clim.nc")

    print()
    print("=" * 60)
    print("Reanalysis II files (full record)")
    print("=" * 60)
    for var in R2_SURFACE_VARIABLES + R2_PRESSURE_VARIABLES:
        validate(r2_folder / f"r2_{var}.mon.mean.nc")
        validate(r2_folder / f"r2_{var}_clim.nc")

    print()
    print("=" * 60)
    print("Reanalysis II files (trimmed to GODAS window)")
    print("=" * 60)
    for var in R2_SURFACE_VARIABLES + R2_PRESSURE_VARIABLES:
        validate(trimmed_folder / f"r2_{var}_trimmed.nc")
        validate(trimmed_folder / f"r2_{var}_trimmed_clim.nc")
