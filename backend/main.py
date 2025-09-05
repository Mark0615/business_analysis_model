# main.py — hardened API (CORS/limits/https/logging/diagnostics/timeouts)
import os, time, uuid, logging, concurrent.futures as futures
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd, numpy as np

# ==== optional: load .env locally ====
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ==== Settings (env vars) ====
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
REQUEST_MAX_BYTES = int(os.getenv("REQUEST_MAX_BYTES", "10485760"))  # 10 MiB
MAX_ROWS = int(os.getenv("MAX_ROWS", "200000"))                      # rows limit per request
ALGO_TIMEOUT_SEC = int(os.getenv("ALGO_TIMEOUT_SEC", "15"))          # per-task timeout
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "0").lower() in ("1","true","yes")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")  # optional

# ==== Logging / Sentry ====
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
if SENTRY_DSN:
    try:
        import sentry_sdk  # pip install sentry-sdk
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.1)
        logging.info("Sentry initialized")
    except Exception as e:
        logging.warning(f"Sentry init failed: {e}")

# ==== ML imports ====
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ==== App & middleware ====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

START_TS = time.time()

# Request ID + latency + basic error catcher
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.rid = rid
    t0 = time.time()
    try:
        resp = await call_next(request)
    except Exception as e:
        logging.exception(f"[{rid}] Unhandled error")
        return JSONResponse({"error": "internal_server_error", "rid": rid}, status_code=500)
    dt = int((time.time() - t0) * 1000)
    resp.headers["X-Request-ID"] = rid
    logging.info(f'[{rid}] {request.method} {request.url.path} -> {resp.status_code} in {dt}ms')
    return resp

# HTTPS enforce (behind proxy like Cloud Run / Render)
@app.middleware("http")
async def https_enforcer(request: Request, call_next):
    if REQUIRE_HTTPS:
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        if proto != "https":
            return JSONResponse({"error": "https_required"}, status_code=400)
    return await call_next(request)

# Security headers
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if REQUIRE_HTTPS:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    return resp

# Body size guard (413)
@app.middleware("http")
async def size_guard(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl and int(cl) > REQUEST_MAX_BYTES:
            return JSONResponse({"error":"request_too_large", "limit_bytes": REQUEST_MAX_BYTES}, status_code=413)
        if not cl:
            body = await request.body()  # Starlette caches it; downstream still can read
            if len(body) > REQUEST_MAX_BYTES:
                return JSONResponse({"error":"request_too_large", "limit_bytes": REQUEST_MAX_BYTES}, status_code=413)
            # re-inject is not needed; request stores the body
    return await call_next(request)

# ==== Schemas ====
class Row(BaseModel):
    date: str
    product_name: str
    product_category: Optional[str] = None
    quantity: Optional[float] = 0
    revenue: float
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    order_id: Optional[str] = None

class AnalyzeIn(BaseModel):
    rows: List[Row]

# ==== Health ====
@app.get("/ping")
def ping():
    return {"status": "ok", "uptime_sec": int(time.time() - START_TS)}

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# ==== Helpers ====
def _assoc_rules(df: pd.DataFrame) -> (list, Dict[str, Any]):
    # build basket_id
    cust_key = df.get("customer_email", pd.Series([""] * len(df))).fillna("").str.strip()
    if not (cust_key.str.len() > 0).any():
        cust_key = df.get("customer_name", pd.Series([""] * len(df))).fillna("").str.strip()
    if df.get("order_id", pd.Series([None]*len(df))).notna().any():
        df["basket_id"] = df["order_id"].fillna("NA").astype(str)
    else:
        df["basket_id"] = cust_key + "_" + df["date"].dt.strftime("%Y-%m-%d")

    tx_series = df.groupby("basket_id")["product_name"].apply(lambda s: sorted(set(s)))
    tx_all = int(len(tx_series))
    tx = [t for t in tx_series.tolist() if len(t) >= 2]

    assoc_rules = []
    reason = None
    try:
        if len(tx) >= 5:
            te = TransactionEncoder()
            tx_df = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
            freq = apriori(tx_df, min_support=0.005, use_colnames=True)
            if not freq.empty:
                rules = association_rules(freq, metric="lift", min_threshold=1.05)
                if not rules.empty:
                    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(5)
                    for _, r in rules.iterrows():
                        assoc_rules.append({
                            "antecedents": sorted(list(r["antecedents"])),
                            "consequents": sorted(list(r["consequents"])),
                            "support": round(float(r["support"]), 4),
                            "confidence": round(float(r["confidence"]), 4),
                            "lift": round(float(r["lift"]), 4),
                        })
                else:
                    reason = "no_significant_rules"
            else:
                reason = "too_sparse_support"
        else:
            reason = "single_item_baskets_or_too_few"
    except Exception:
        reason = "algorithm_error"

    diag = {"ok": len(assoc_rules) > 0, "baskets_total": tx_all, "baskets_valid": int(len(tx)), "reason": reason}
    return assoc_rules, diag

def _rfm(df: pd.DataFrame) -> (Dict[str,float], Dict[str,int], Dict[str,float], Dict[str, list], Dict[str,Any]):
    # customer key
    cust_key = df.get("customer_email", pd.Series([""] * len(df))).fillna("").str.strip()
    if not (cust_key.str.len() > 0).any():
        cust_key = df.get("customer_name", pd.Series([""] * len(df))).fillna("").str.strip()
    df = df.join(cust_key.rename("customer_id"))
    if df.get("order_id", pd.Series([None]*len(df))).notna().any():
        df["basket_id"] = df["order_id"].fillna("NA").astype(str)
    else:
        df["basket_id"] = df["customer_id"].fillna("") + "_" + df["date"].dt.strftime("%Y-%m-%d")
    df["order_key"] = df["basket_id"]

    cust = (df.groupby("customer_id")
              .agg(last_date=("date", "max"),
                   orders=("order_key", pd.Series.nunique),
                   monetary=("revenue", "sum"))
              .reset_index())
    cust = cust[cust["customer_id"].fillna("").str.len() > 0]
    customers_total = int(df["customer_id"].fillna("").str.len().gt(0).sum())
    customers_used = int(len(cust))

    rfm_share = {"low":0.0,"mid":0.0,"high":0.0}
    rfm_counts = {"low":0,"mid":0,"high":0}
    rfm_avg_monetary = {"low":0.0,"mid":0.0,"high":0.0}
    nurture = {"mid": [], "low": []}
    reason = None

    if len(cust) >= 3:
        max_date = df["date"].max()
        cust["recency"] = (max_date - cust["last_date"]).dt.days

        # 先嘗試 KMeans（標準化）
        X = cust[["recency","orders","monetary"]]
        X = (X - X.mean()) / (X.std(ddof=0) + 1e-9)
        use_quantile = False
        try:
            km = KMeans(n_clusters=3, n_init=10, random_state=42)
            cust["segment"] = km.fit_predict(X)
            rank = (cust.groupby("segment")["monetary"].mean().sort_values().reset_index())
            rank["level"] = ["low","mid","high"]
            m = dict(zip(rank["segment"], rank["level"]))
            cust["level"] = cust["segment"].map(m)

            counts = cust["segment"].value_counts()
            # 只有在群數 <3 或某群極端過少才回退
            if (len(counts) < 3) or (counts.min() <= 1):
                use_quantile = True
        except Exception:
            use_quantile = True

        if use_quantile:
            # 用 monetary 三分位，讓三群更平均
            q1, q2 = cust["monetary"].quantile([1/3, 2/3])
            def lvl(v):
                if v <= q1: return "low"
                elif v <= q2: return "mid"
                else: return "high"
            cust["level"] = cust["monetary"].apply(lvl)

        total_rev = cust["monetary"].sum()
        for lv in ["low","mid","high"]:
            rfm_counts[lv] = int((cust["level"]==lv).sum())
            avg = float(cust.loc[cust["level"]==lv,"monetary"].mean() or 0.0)
            rfm_avg_monetary[lv] = round(avg, 2)
            part = float(cust.loc[cust["level"]==lv,"monetary"].sum())
            rfm_share[lv] = round(part/total_rev, 4) if total_rev>0 else 0.0

        # 加溫名單：以「最近沒回購且有潛力」為主；中/低兩群分開輸出
        r_th = 45
        m_th_mid = cust.loc[cust["level"]=="mid","monetary"].median() if (cust["level"]=="mid").any() else 0.0
        m_th_low = cust.loc[cust["level"]=="low","monetary"].median() if (cust["level"]=="low").any() else 0.0

        def build_list(df_seg, m_th):
            seg = df_seg[(df_seg["recency"]>r_th) & (df_seg["monetary"]>=m_th)]
            out = (seg.sort_values("monetary", ascending=False)
                      .head(200)[["customer_id","last_date","orders","monetary"]]
                      .rename(columns={"customer_id":"customer_email",
                                       "last_date":"last_order_date",
                                       "orders":"lifetime_orders",
                                       "monetary":"lifetime_revenue"})
                      .to_dict(orient="records"))
            return out

        nurture["mid"] = build_list(cust[cust["level"]=="mid"], m_th_mid)
        nurture["low"] = build_list(cust[cust["level"]=="low"], m_th_low)

    else:
        reason = "no_customer_key_or_too_few"

    diag = {"ok": customers_used >= 3, "customers_total": customers_total, "customers_used": customers_used, "reason": reason}
    return rfm_share, rfm_counts, rfm_avg_monetary, nurture, diag

def _potential_products(df: pd.DataFrame, top5_names: set, monthly_ordered: pd.DataFrame) -> list:
    out = []
    df["yyyymm"] = df["date"].dt.to_period("M").astype(str)
    prod_m = (df.groupby(["product_name","yyyymm"], as_index=False)["revenue"].sum()
                .sort_values(["product_name","yyyymm"]))
    if not prod_m.empty and len(monthly_ordered) >= 3:
        last3 = sorted(monthly_ordered["yyyymm"].unique())[-3:]
        recent = prod_m[prod_m["yyyymm"].isin(last3)]
        if not recent.empty:
            g = recent.pivot(index="product_name", columns="yyyymm", values="revenue").fillna(0.0)
            if g.shape[1] == 3:
                g["mom_last"] = (g.iloc[:,2] - g.iloc[:,1]) / (g.iloc[:,1].replace(0, np.nan))
                g["sum3"] = g.sum(axis=1)
                cand = g[(g["mom_last"] > 0.2) & (~g.index.isin(top5_names))]
                cand = cand.sort_values(["mom_last","sum3"], ascending=False).head(5)
                for name, row in cand.iterrows():
                    out.append({"product_name": name,
                                "mom_last": round(float(row["mom_last"]), 3),
                                "sum3_revenue": round(float(row["sum3"]), 2)})
    return out

# ==== API ====
@app.post("/analyze")
async def analyze(payload: AnalyzeIn, request: Request):
    rid = getattr(request.state, "rid", "-")

    # Row count guard
    n = len(payload.rows or [])
    if n == 0:
        return {"error":"no_rows", "diagnostics":{"required":{"ok":False,"reason":"empty_payload"}}}
    if n > MAX_ROWS:
        raise HTTPException(status_code=422, detail={"error":"too_many_rows","max_rows":MAX_ROWS,"got":n})

    # Build DataFrame + basic clean
    df = pd.DataFrame([r.dict() for r in payload.rows])
    required_cols = ["date","product_name","revenue"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {"error":"missing_required_fields",
                "diagnostics":{"required":{"ok":False,"reason":"missing_fields","missing":missing}}}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","product_name"])
    df["quantity"] = pd.to_numeric(df.get("quantity",0), errors="coerce").fillna(0)
    df["revenue"]  = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    if df.empty:
        return {"error":"invalid_required_values",
                "diagnostics":{"required":{"ok":False,"reason":"all_rows_invalid"}}}

    # Top5 / Monthly / MoM / WoW
    top5_df = (df.groupby("product_name", as_index=False)["revenue"].sum()
                 .sort_values("revenue", ascending=False).head(5))
    top5 = top5_df.to_dict(orient="records")

    monthly_df = (df.assign(yyyymm=df["date"].dt.strftime("%Y-%m"))
                    .groupby("yyyymm", as_index=False)["revenue"].sum()
                    .sort_values("yyyymm"))
    mom = None
    if len(monthly_df) >= 2:
        a, b = monthly_df.iloc[-2]["revenue"], monthly_df.iloc[-1]["revenue"]
        mom = None if a == 0 else (b - a) / a

    last_day = df["date"].max()
    r1 = df.loc[(df["date"] >= last_day - pd.Timedelta(days=13)) & (df["date"] < last_day - pd.Timedelta(days=6)), "revenue"].sum()
    r2 = df.loc[(df["date"] >= last_day - pd.Timedelta(days=6)) & (df["date"] <= last_day), "revenue"].sum()
    wow = None if r1 == 0 else (r2 - r1) / r1

    # Run assoc + rfm with timeouts
    assoc_rules, assoc_diag = [], {"ok":False,"reason":"timeout","baskets_total":0,"baskets_valid":0}
    rfm_share = {"low":0.0,"mid":0.0,"high":0.0}
    rfm_counts = {"low":0,"mid":0,"high":0}
    rfm_avg_monetary = {"low":0.0,"mid":0.0,"high":0.0}
    nurture_list = []
    with futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_assoc = ex.submit(_assoc_rules, df.copy())
        f_rfm   = ex.submit(_rfm, df.copy())
        try:
            assoc_rules, assoc_diag = f_assoc.result(timeout=ALGO_TIMEOUT_SEC)
        except futures.TimeoutError:
            assoc_diag = {"ok":False,"reason":"timeout","baskets_total":0,"baskets_valid":0}
            logging.warning(f"[{rid}] assoc timeout")
        try:
            rfm_share, rfm_counts, rfm_avg_monetary, nurture_list, rfm_diag = f_rfm.result(timeout=ALGO_TIMEOUT_SEC)
        except futures.TimeoutError:
            rfm_diag = {"ok":False,"reason":"timeout","customers_total":0,"customers_used":0}
            logging.warning(f"[{rid}] rfm timeout")

    potential_products = _potential_products(df.copy(), set(top5_df["product_name"].tolist()), monthly_df)

    # rfm_* 在上面
    nurture_mid = nurture_low = []
    try:
        rfm_share, rfm_counts, rfm_avg_monetary, nurture_dict, rfm_diag = f_rfm.result(timeout=ALGO_TIMEOUT_SEC)
        nurture_mid = nurture_dict.get("mid", [])
        nurture_low = nurture_dict.get("low", [])
    except futures.TimeoutError:
        rfm_diag = {"ok":False,"reason":"timeout","customers_total":0,"customers_used":0}
        logging.warning(f"[{rid}] rfm timeout")

    return {
        "top5_products": top5,
        "monthly": monthly_df.to_dict(orient="records"),
        "mom": mom, "wow": wow,
        "assoc_rules": assoc_rules,
        "rfm_share": rfm_share,
        "rfm_counts": rfm_counts,
        "rfm_avg_monetary": rfm_avg_monetary,
        "nurture_list": (nurture_mid + nurture_low),   # 向下相容
        "nurture_mid": nurture_mid,
        "nurture_low": nurture_low,
        "potential_products": potential_products,
        "diagnostics": {"assoc": assoc_diag, "rfm": rfm_diag, "required": {"ok": True}},
    }
