"""scheduler.py — the nightly pipeline + email alerts.

Run as a long-lived process (systemd / Docker) or wire each job to cron.
    python scheduler.py
"""
import os, datetime as dt
from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from ingest import fetch_weather, to_weekly, fetch_weather_archive
from predict import forecast_area, band
import email_worker

load_dotenv()
engine = create_engine(os.environ.get("DATABASE_URL", "sqlite:///prohori.db"))
MED, HIGH, EXT = (int(os.environ.get(k, d)) for k, d in
                  [("ALERT_MEDIUM", 147), ("ALERT_HIGH", 374), ("ALERT_EXTREME", 800)])

def daily_job():
    today = dt.date.today()
    print(f"[{today}] running daily pipeline")
    with engine.connect() as c:
        areas = c.execute(text("SELECT id,lat,lon FROM areas")).mappings().all()

    for a in areas:
        # 1) ingest fresh weather  2) (weekly) NDWI from gee_ndwi  3) predict
        weekly = to_weekly(fetch_weather(a["lat"], a["lon"]))
        weekly["area_id"] = a["id"]
        for col, v in {"ndwi": -0.1, "urban_index": 0.8, "population_index": 0.8, "cases": 0}.items():
            weekly[col] = v   # TODO: join real gee_indices + dengue_cases
        fc = forecast_area(weekly)

        with engine.begin() as c:
            for h, p in fc["preds"].items():
                c.execute(text("""INSERT INTO forecasts(area_id,run_date,horizon_wk,pred,lo,hi,risk_pct,band,drivers)
                    VALUES(:a,:d,:h,:p,:lo,:hi,:pct,:b,:dr)
                    ON CONFLICT(area_id,run_date,horizon_wk) DO UPDATE
                    SET pred=:p,lo=:lo,hi=:hi,risk_pct=:pct,band=:b,drivers=:dr"""),
                    {"a": a["id"], "d": today, "h": h, "p": p["pred"], "lo": p["lo"],
                     "hi": p["hi"], "pct": fc["risk_pct"], "b": fc["band"],
                     "dr": str(fc["drivers"])})

        # 4) early-warning: alert subscribers if band rose to mod/high/ext
        maybe_alert(a["id"], fc)

def maybe_alert(area_id, fc):
    new_band = fc["band"]
    if new_band == "low":
        return
    with engine.begin() as c:
        subs = c.execute(text("""SELECT id,email,last_band FROM subscriptions
                                 WHERE area_id=:a AND verified=true"""),
                         {"a": area_id}).mappings().all()
        for s in subs:
            if s["last_band"] != new_band:        # only on change -> no spam
                email_worker.send_alert(s["email"], area_id, fc)
                c.execute(text("UPDATE subscriptions SET last_band=:b WHERE id=:i"),
                          {"b": new_band, "i": s["id"]})

if __name__ == "__main__":
    sched = BlockingScheduler(timezone="Asia/Dhaka")
    sched.add_job(daily_job, "cron", hour=6, minute=0)   # 6 AM daily
    print("scheduler started — Ctrl+C to stop")
    daily_job()          # run once on startup
    sched.start()
