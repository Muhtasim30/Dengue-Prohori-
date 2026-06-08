"""gee_ndwi.py — pull weekly NDWI (+ nighttime-lights urban index, population)
from Google Earth Engine for each area, and write to the gee_indices table.

Setup (one time):
  1. Create a Google Cloud project, enable the Earth Engine API.
  2. Create a service account, grant it Earth Engine access, download key JSON.
  3. pip install earthengine-api ; set GEE_SERVICE_ACCOUNT and GEE_KEY_FILE in .env
Run weekly (e.g. from scheduler.py):
  python gee_ndwi.py
"""
import os, datetime as dt
import ee
from dotenv import load_dotenv
load_dotenv()

def init():
    creds = ee.ServiceAccountCredentials(os.environ["GEE_SERVICE_ACCOUNT"],
                                          os.environ["GEE_KEY_FILE"])
    ee.Initialize(creds)

def ndwi_for(lat, lon, start, end):
    """Mean NDWI over a small buffer around the area for the week."""
    pt = ee.Geometry.Point([lon, lat]).buffer(3000)   # ~3 km radius
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .filterBounds(pt).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUD_COVER", 20)))
    def ndwi(img):
        g = img.select("SR_B3"); nir = img.select("SR_B5")
        return g.subtract(nir).divide(g.add(nir)).rename("NDWI")
    if col.size().getInfo() == 0:
        return None
    img = col.map(ndwi).mean()
    val = img.reduceRegion(ee.Reducer.mean(), pt, 30).get("NDWI").getInfo()
    return val

def urban_index(lat, lon, start, end):
    """VIIRS nighttime lights -> proxy for built-up intensity (0..1-ish)."""
    pt = ee.Geometry.Point([lon, lat]).buffer(3000)
    col = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
           .filterBounds(pt).filterDate(start, end).select("avg_rad"))
    if col.size().getInfo() == 0:
        return None
    v = col.mean().reduceRegion(ee.Reducer.mean(), pt, 500).get("avg_rad").getInfo()
    return v

def run(areas, week_start):
    """areas: list of dicts {id,lat,lon}. Returns rows to upsert into gee_indices."""
    init()
    start = week_start.isoformat()
    end = (week_start + dt.timedelta(days=7)).isoformat()
    rows = []
    for a in areas:
        rows.append({
            "area_id": a["id"], "week_start": week_start,
            "ndwi": ndwi_for(a["lat"], a["lon"], start, end),
            "urban_index": urban_index(a["lat"], a["lon"], start, end),
        })
        print("NDWI", a["id"], rows[-1]["ndwi"])
    return rows

if __name__ == "__main__":
    # demo for Dhaka centre, last full week
    today = dt.date.today()
    monday = today - dt.timedelta(days=today.weekday()+7)
    print(run([{"id": "dhaka", "lat": 23.81, "lon": 90.41}], monday))
