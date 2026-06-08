// frontend_integration.js
// How to switch the HTML app from "compute-in-browser" to "use my backend".
// In dengue-prohori.html, the forecast is produced by summaryFor(daily, area).
// To use the real trained model on your server, replace the data path like this:

const API = "https://api.yourdomain.org";   // your FastAPI base URL

// 1) Forecast for the selected area (replaces local summaryFor for the headline)
async function apiForecast(area) {
  const r = await fetch(`${API}/forecast?area_id=${area.id}&lat=${area.lat}&lon=${area.lon}`);
  return r.json();   // { band, risk_pct, wk1, preds:{1:{pred,lo,hi},...}, drivers:{...} }
}

// 2) Heatmap: every area under the district, computed server-side from the model
async function apiHeatmap(parent) {
  return (await fetch(`${API}/heatmap?parent=${parent}`)).json();
}

// 3) Subscribe (real email, double opt-in handled by the backend)
async function apiSubscribe(email, areaId) {
  return (await fetch(`${API}/subscribe`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ email, area_id: areaId })
  })).json();
}

// 4) Community breeding report (with uploaded photo URL from object storage)
async function apiReport(rep) {
  return (await fetch(`${API}/report`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(rep)
  })).json();
}

// 5) Admin notices (token kept server-side / in an admin-only build)
async function apiPostNotice(notice, adminToken) {
  return (await fetch(`${API}/notices`, {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-Admin-Token": adminToken},
    body: JSON.stringify(notice)
  })).json();
}

/* MIGRATION NOTES
 * - Keep the browser engine as an offline fallback; try the API first.
 * - The model "drivers" object feeds the existing glass-box "Why?" card directly.
 * - For the LSTM, export to TensorFlow.js and run client-side, OR serve from /forecast.
 * - Photos: upload to S3/GCS via a presigned URL, then send the photo_url to /report.
 */
