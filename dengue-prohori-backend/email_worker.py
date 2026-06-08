"""email_worker.py — send verification + early-warning emails via SendGrid.
Swap for Amazon SES or SMTP if preferred. SMS: add Twilio/local gateway here.
"""
import os
from dotenv import load_dotenv
load_dotenv()

FROM = os.environ.get("ALERT_FROM_EMAIL", "alerts@example.org")
BASE = os.environ.get("PUBLIC_BASE_URL", "https://example.org")

def _send(to, subject, html):
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        msg = Mail(from_email=FROM, to_emails=to, subject=subject, html_content=html)
        SendGridAPIClient(os.environ["SENDGRID_API_KEY"]).send(msg)
        print("sent ->", to)
    except Exception as e:
        print("email error:", e)        # log; don't crash the pipeline

def send_verification(email, token):
    link = f"{BASE}/verify?token={token}"
    _send(email, "Confirm your Dengue Prohori alerts",
          f"<p>Confirm your subscription:</p><p><a href='{link}'>{link}</a></p>")

def send_alert(email, area_id, fc):
    p = fc["preds"]["1"] if "1" in fc["preds"] else list(fc["preds"].values())[0]
    band = fc["band"].upper()
    _send(email, f"Dengue alert — {area_id}: {band} risk",
          f"""<h2>Dengue risk in {area_id}: {band} ({fc['risk_pct']}%)</h2>
              <p><b>Now:</b> ~{p['pred']} cases expected this week (range {p['lo']}–{p['hi']}).</p>
              <p><b>What to do:</b> Remove standing water, use nets/repellent, and watch for fever.</p>
              <p><a href='{BASE}'>Open Dengue Prohori</a></p>""")
