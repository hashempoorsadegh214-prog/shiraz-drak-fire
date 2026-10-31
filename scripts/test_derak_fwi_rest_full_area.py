import csv
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import requests
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_FILE = ROOT / "Derak_Mountain.geojson"
DATA_DIR = ROOT / "data"

JSON_OUTPUT = DATA_DIR / "derak_fwi_forecast.json"
CSV_OUTPUT = DATA_DIR / "derak_fwi_forecast.csv"
POINTS_OUTPUT = DATA_DIR / "derak_fwi_forecast_points.csv"
ERRORS_OUTPUT = DATA_DIR / "derak_fwi_forecast_errors.csv"

API_URL = (
    "https://api.effis.emergency.copernicus.eu/"
    "rest/2/burntareas/charts/wms"
)

GRID_STEP = 0.008
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
TIMEZONE = ZoneInfo("Asia/Tehran")


def now_tehran():
    return datetime.now(TIMEZONE)


def parse_target_date():
    raw = os.environ.get("TARGET_DATE", "").strip()

    if not raw:
        return now_tehran().date()

    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            "TARGET_DATE must use YYYY-MM-DD format."
        ) from exc


def parse_days_ahead():
    raw = os.environ.get("DAYS_AHEAD", "10").strip()

    try:
        days = int(raw)
    except ValueError as exc:
        raise ValueError(
            "DAYS_AHEAD must be an integer."
        ) from exc

    if not 0 <= days <= 14:
        raise ValueError(
            "DAYS_AHEAD must be between 0 and 14."
        )

    return days


def load_boundary():
    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {BOUNDARY_FILE}"
        )

    gdf = gpd.read_file(BOUNDARY_FILE)

    if gdf.empty:
        raise ValueError("Boundary file is empty.")

    if gdf.crs is None:
        raise ValueError(
            "Boundary CRS is missing. "
            "Set the correct CRS in the GeoJSON before continuing."
        )

    gdf = gdf.to_crs("EPSG:4326")
    boundary = gdf.geometry.union_all()

    if boundary.is_empty:
        raise ValueError("Boundary geometry is empty.")

    if not boundary.is_valid:
        boundary = boundary.buffer(0)

    if boundary.is_empty or not boundary.is_valid:
        raise ValueError("Boundary geometry is invalid.")

    return boundary


def create_grid_points(boundary):
    min_lon, min_lat, max_lon, max_lat = boundary.bounds

    points = []
    lat = min_lat

    while lat <= max_lat + 1e-12:
        lon = min_lon

        while lon <= max_lon + 1e-12:
            point = Point(lon, lat)

            if boundary.covers(point):
                points.append(
                    {
                        "point_id": len(points) + 1,
                        "longitude": round(lon, 8),
                        "latitude": round(lat, 8),
                    }
                )

            lon += GRID_STEP

        lat += GRID_STEP

    if not points:
        raise ValueError(
            "No grid points were generated inside the boundary."
        )

    return points


def as_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        try:
            parsed = json.loads(stripped)

            if isinstance(parsed, list):
                return parsed

            return [parsed]
        except json.JSONDecodeError:
            return [value]

    return [value]


def parse_date(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(
                value, tz=TIMEZONE
            ).date().isoformat()
        except (ValueError, OSError, OverflowError):
            return None

    text = str(value).strip()

    if not text:
        return None

    # Accept ISO dates and ISO timestamps.
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        pass

    # Accept common date labels.
    for fmt in (
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    return None


def parse_numeric(value):
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def extract_series(payload):
    """
    Extract date/FWI pairs from the REST response.

    The response is inspected without assuming that every API
    response has exactly the same JSON wrapper.
    """

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []

    if isinstance(payload, list):
        output = []

        for item in payload:
            output.extend(extract_series(item))

        return output

    if not isinstance(payload, dict):
        return []

    output = []

    x_data = payload.get("x_data")
    y_data = payload.get("y_data")

    if x_data is not None and y_data is not None:
        dates = as_list(x_data)

        if isinstance(y_data, dict):
            fwi_values = as_list(y_data.get("fwi"))
        elif isinstance(y_data, list):
            # Some responses may return an array of objects.
            fwi_values = []
            for item in y_data:
                if isinstance(item, dict):
                    fwi_values.append(item.get("fwi"))
                else:
                    fwi_values.append(item)
        else:
            fwi_values = as_list(y_data)

        for raw_date, raw_value in zip(dates, fwi_values):
            parsed_date = parse_date(raw_date)
            value = parse_numeric(raw_value)

            if parsed_date and value is not None:
                output.append(
                    {
                        "date": parsed_date,
                        "fwi": value,
                    }
                )

    # Support nested response structures.
    for key, value in payload.items():
        if key in {"x_data", "y_data"}:
            continue

        if isinstance(value, (dict, list)):
            output.extend(extract_series(value))

    # Remove exact duplicate pairs.
    unique = {}
    for item in output:
        unique[(item["date"], item["fwi"])] = item

    return list(unique.values())


def request_point_series(session, point, start_date, end_date):
    params = {
        "model": "ecmwf",
        "day_gte": start_date.isoformat(),
        "day_lte": end_date.isoformat(),
        "point": (
            f"({point['longitude']} {point['latitude']})"
        ),
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()
            payload = response.json()

            series = extract_series(payload)

            # Only keep values inside the requested date range.
            filtered = []

            for item in series:
                item_date = date.fromisoformat(item["date"])

                if start_date <= item_date <= end_date:
                    filtered.append(item)

            return filtered, None

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)

    return [], last_error


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_date = parse_target_date()
    days_ahead = parse_days_ahead()
    end_date = start_date + timedelta(days=days_ahead)

    print(f"Tehran time: {now_tehran().isoformat()}")
    print(f"Requested start date: {start_date}")
    print(f"Requested end date:   {end_date}")

    boundary = load_boundary()
    points = create_grid_points(boundary)

    print(f"Grid spacing: {GRID_STEP} degrees")
    print(f"Grid points inside boundary: {len(points)}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Derak-FWI-Forecast-Monitor/1.0"
            ),
            "Accept": "application/json",
        }
    )

    observations = []
    errors = []

    for index, point in enumerate(points, start=1):
        print(
            f"[{index}/{len(points)}] "
            f"Point {point['point_id']}: "
            f"{point['longitude']}, {point['latitude']}"
        )

        series, error = request_point_series(
            session,
            point,
            start_date,
            end_date,
        )

        if error:
            errors.append(
                {
                    "point_id": point["point_id"],
                    "longitude": point["longitude"],
                    "latitude": point["latitude"],
                    "error": error,
                }
            )

        if not series and not error:
            errors.append(
                {
                    "point_id": point["point_id"],
                    "longitude": point["longitude"],
                    "latitude": point["latitude"],
                    "error": (
                        "No valid date/FWI pairs returned "
                        "for the requested interval."
                    ),
                }
            )

        for item in series:
            observations.append(
                {
                    "date": item["date"],
                    "point_id": point["point_id"],
                    "longitude": point["longitude"],
                    "latitude": point["latitude"],
                    "fwi": item["fwi"],
                }
            )

        # Keep requests polite and reduce the chance of rate limiting.
        time.sleep(0.15)

    requested_dates = [
        (start_date + timedelta(days=offset)).isoformat()
        for offset in range(days_ahead + 1)
    ]

    rows = []

    for forecast_date in requested_dates:
        values = [
            item["fwi"]
            for item in observations
            if item["date"] == forecast_date
        ]

        unique_point_ids = {
            item["point_id"]
            for item in observations
            if item["date"] == forecast_date
        }

        count = len(unique_point_ids)
        expected = len(points)

        if values:
            mean_value = sum(values) / len(values)
            minimum = min(values)
            maximum = max(values)
        else:
            mean_value = None
            minimum = None
            maximum = None

        rows.append(
            {
                "date": forecast_date,
                "expected_points": expected,
                "valid_points": count,
                "coverage_percent": round(
                    count / expected * 100, 2
                ) if expected else 0,
                "mean_fwi": round(mean_value, 6)
                if mean_value is not None else None,
                "min_fwi": round(minimum, 6)
                if minimum is not None else None,
                "max_fwi": round(maximum, 6)
                if maximum is not None else None,
                "complete": count == expected,
            }
        )

    write_csv(
        CSV_OUTPUT,
        rows,
        [
            "date",
            "expected_points",
            "valid_points",
            "coverage_percent",
            "mean_fwi",
            "min_fwi",
            "max_fwi",
            "complete",
        ],
    )

    write_csv(
        POINTS_OUTPUT,
        observations,
        [
            "date",
            "point_id",
            "longitude",
            "latitude",
            "fwi",
        ],
    )

    write_csv(
        ERRORS_OUTPUT,
        errors,
        [
            "point_id",
            "longitude",
            "latitude",
            "error",
        ],
    )

    report = {
        "generated_at_tehran": now_tehran().isoformat(),
        "source": API_URL,
        "model": "ecmwf",
        "boundary_file": BOUNDARY_FILE.name,
        "grid_step_degrees": GRID_STEP,
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "expected_points": len(points),
        "successful_observation_count": len(observations),
        "failed_point_requests": len(errors),
        "date_summaries": rows,
        "observations": observations,
        "errors": errors,
        "notes": [
            (
                "Only dates explicitly returned by the API "
                "are represented as valid observations."
            ),
            (
                "A daily mean is complete only when all "
                "generated grid points have valid data for that date."
            ),
            (
                "This is a test/archive workflow. It does not "
                "change the live website's displayed FWI."
            ),
        ],
    }

    with JSON_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDAILY SUMMARY")
    print("=" * 78)

    for row in rows:
        print(
            f"{row['date']} | "
            f"{row['valid_points']}/{row['expected_points']} points | "
            f"coverage={row['coverage_percent']}% | "
            f"mean={row['mean_fwi']} | "
            f"min={row['min_fwi']} | "
            f"max={row['max_fwi']} | "
            f"complete={row['complete']}"
        )

    print("=" * 78)
    print(f"JSON: {JSON_OUTPUT}")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"Point records: {POINTS_OUTPUT}")
    print(f"Errors: {ERRORS_OUTPUT}")

    # Do not report success if the requested current date is incomplete.
    today_row = next(
        (
            row for row in rows
            if row["date"] == start_date.isoformat()
        ),
        None,
    )

    if today_row is None or not today_row["complete"]:
        print(
            "\nERROR: The start date does not have complete "
            "coverage across all grid points."
        )
        sys.exit(1)

    print(
        "\nSUCCESS: The start date has complete coverage "
        "across all grid points."
    )


if __name__ == "__main__":
    main()
