"""main.py — FastAPI backend for Dengue Prohori.

Run:  uvicorn main:app --reload --port 8000
The HTML frontend calls these endpoints instead of computing locally.
"""
import os, hashlib, secrets, datetime as dt
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd

from ingest import fetch_weather, to_weekly
from predict import forecast_area

load_dotenv()
engine = create_engine(os.environ.get("DATABASE_URL", "sqlite:///prohori.db"))
ADMIN = os.environ.get("ADMIN_TOKEN", "dev")

app = FastAPI(title="Dengue Prohori API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- forecasts ----------
@app.get("/forecast")
def forecast(area_id: str = "dhaka", lat: float = 23.81, lon: float = 90.41):
    """Live forecast for one area. In production, read the latest precomputed row
    from the `forecasts` table (written nightly). Here we compute on the fly."""
    weekly = to_weekly(fetch_weather(lat, lon))
    weekly["area_id"] = area_id
    # NDWI/urban/population would be joined from gee_indices; defaults if absent:
    for c, v in {"ndwi": -0.1, "urban_index": 0.8, "population_index": 0.8, "cases": 0}.items():
        weekly[c] = v
    return {"area_id": area_id, **forecast_area(weekly)}

@app.get("/heatmap")
def heatmap(parent: str = "Dhaka"):
    """Return risk for every area under a district (drive the map)."""
    with engine.connect() as c:
        areas = c.execute(text("SELECT id,lat,lon FROM areas WHERE parent=:p"),
                          {"p": parent}).mappings().all()
    out = []
    for a in areas:
        out.append({"area_id": a["id"], **forecast(a["id"], a["lat"], a["lon"])})
    return out

# ---------- subscriptions (double opt-in) ----------
class Sub(BaseModel):
    email: EmailStr
    area_id: str

@app.post("/subscribe")
def subscribe(s: Sub):
    token = secrets.token_urlsafe(24)
    with engine.begin() as c:
        c.execute(text("""INSERT INTO subscriptions(email,area_id,token,verified)
                          VALUES(:e,:a,:t,false)
                          ON CONFLICT(email,area_id) DO UPDATE SET token=:t"""),
                  {"e": s.email, "a": s.area_id, "t": token})
    # email_worker.send_verification(s.email, token)  <-- enable in production
    return {"ok": True, "message": "Check your email to confirm."}

@app.get("/verify")
def verify(token: str):
    with engine.begin() as c:
        c.execute(text("UPDATE subscriptions SET verified=true,token=null WHERE token=:t"),
                  {"t": token})
    return {"ok": True}

# ---------- community reports ----------
class Report(BaseModel):
    type: str; description: str = ""; address: str = ""
    lat: float; lon: float; area_id: str = ""; photo_url: str = ""

@app.post("/report")
def report(r: Report):
    with engine.begin() as c:
        c.execute(text("""INSERT INTO reports(type,description,address,lat,lon,geom,photo_url,area_id)
                          VALUES(:t,:d,:ad,:la,:lo,ST_MakePoint(:lo,:la)::geography,:ph,:ar)"""),
                  {"t": r.type, "d": r.description, "ad": r.address, "la": r.lat,
                   "lo": r.lon, "ph": r.photo_url, "ar": r.area_id})
    return {"ok": True, "status": "pending"}

@app.get("/hotspots")
def hotspots(parent: str = "Dhaka"):
    with engine.connect() as c:
        rows = c.execute(text("""SELECT type,lat,lon,description FROM reports
                                 WHERE status='verified' AND area_id=:p"""),
                        {"p": parent}).mappings().all()
    return [dict(r) for r in rows]

# ---------- patient case (sensitive) ----------
class Case(BaseModel):
    area_id: str; method: str; status: str
    nid: str = ""          # hashed, never stored raw
    consented: bool

@app.post("/case")
def case(c_in: Case):
    if not c_in.consented:
        raise HTTPException(400, "consent required")
    nid_hash = hashlib.sha256(c_in.nid.encode()).hexdigest() if c_in.nid else None
    with engine.begin() as c:
        c.execute(text("""INSERT INTO patient_cases(area_id,method,status,nid_hash,consented)
                          VALUES(:a,:m,:s,:h,true)"""),
                  {"a": c_in.area_id, "m": c_in.method, "s": c_in.status, "h": nid_hash})
    return {"ok": True}

# ---------- notices (admin) ----------
class Notice(BaseModel):
    level: str; title: str; body: str

@app.get("/notices")
def notices():
    with engine.connect() as c:
        rows = c.execute(text("SELECT level,title,body FROM notices WHERE active ORDER BY created_at DESC")).mappings().all()
    return [dict(r) for r in rows]

@app.post("/notices")
def post_notice(n: Notice, x_admin_token: str = Header(default="")):
    if x_admin_token != ADMIN:
        raise HTTPException(403, "admin only")
    with engine.begin() as c:
        c.execute(text("INSERT INTO notices(level,title,body) VALUES(:l,:t,:b)"),
                  {"l": n.level, "t": n.title, "b": n.body})
    return {"ok": True}
