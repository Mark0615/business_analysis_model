# main.py — Hardened FastAPI backend with CORS, limits, timeouts, and diagnostics
import os, re, time, uuid, logging
import concurrent.futures as futures
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============== optional: load local .env for dev ==============
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ========================== Settings ===========================
CORS_ORIGINS_RAW = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
)

# Starlette 的 CORSMiddleware 對萬用字元不吃 "https://*.xxx"
# 這裡將帶 * 的項目轉成 allow_origin_regex，其它維持精確比對。
_raw_origins = [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]
_plain_origins = [o for o in _raw_origins if "*" not in o]
_wildcards = [o for o in _raw_origins if "*" in o]
if _wildcards:
    _patterns = [re.escape(w).replace(r"\*", r"[^/]+") for w in _wildcards]
    CORS_ORIGIN_REGEX = r"^(?:%s)$" % "|".join(_patterns)
else:
    CORS_ORIGIN_REGEX = None

REQUEST_MAX_BYTES = int(os.getenv("REQUEST_MAX_BYTES", "10485760"))  # 10 MiB
MAX_ROWS = int(os.getenv("MAX_ROWS", "200000"))                      # upper row limit
ALGO_TIMEOUT_SEC = int(os.getenv("ALGO_TIMEOUT_SEC", "20"))          # per-task timeout
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "0").lower() in ("1", "true", "yes")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")  # optional

# ====================== Logging / Sentry =======================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)

if SENTRY_DSN:
    try:
        import sentry_sdk  # pip install sentry-sdk
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.05)
        logging.info("Sentry initialized")
    except Exception as e:
        logging.warning(f"Sentry init failed: {e}")

# ================ Lazy ML imports (faster cold start) =========
def _lazy_import_ml():
    # 只在需要時才 import heavy 套件
    from sklearn.cluster import KMeans
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    return KMeans, TransactionEncoder, apriori, association_rules

# ======================== FastAPI app ==========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_plain_origins,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

START_TS = time.time()

# --------------------- middleware: request id / logging
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.rid = rid
    t0 = time.time()
    try:
        resp = await call_next(request)
    except Exception:
        logging.exception(f"[{rid}] unhandled error")
        return JSONResponse({"error": "internal_server_error", "rid": rid}, status_code=500)
    dt_ms = int((time.time() - t0) * 1000)
    resp.headers["X-Request-ID"] = rid
    logging.info(f"[{rid}] {request.method} {request.url.path} -> {resp.status_code} in {dt_ms}ms")
    return resp

# --------------------- middleware: HTTPS enforce
@app.middleware("http")
async def https_enforcer(request: Request, call_next):
    if REQUIRE_HTTPS:
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        if proto != "https":
            return JSONResponse({"error": "https_required"}, status_code=400)
    return await call_next(request)

# --------------------- middleware: security headers
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if REQUIRE_HTTPS:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    return resp

# --------------------- middleware: request body size guard
@app.middleware("http")
async def size_guard(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl and int(cl) > REQUEST_MAX_BYTES:
            return JSONResponse({"error": "request_too_large", "limit_bytes": REQUEST_MAX_BYTES}, status_code=413)
        if not cl:
            body = await request.body()  # Starlette caches it
            if len(body) > REQUEST_MAX_BYTES:
                return JSONResponse({"error": "request_too_large", "limit_bytes": REQUEST_MAX_BYTES}, status_code=413)
    return await call_next(request)

# ======================== Schemas ==============================
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
    rows: List[Row]  # 👈 只要 rows，別放 date_from/date_to

class AnalyzeAssocIn(BaseModel):
    rows: List[Row]
    date_from: str   # 'YYYY-MM-DD'
    date_to: str     # 'YYYY-MM-DD'


# ======================== Health ===============================
@app.get("/ping")
def ping():
    return {"status": "ok", "uptime_sec": int(time.time() - START_TS)}

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# ======================== Helpers ==============================
def _assoc_rules(df: pd.DataFrame) -> Tuple[list, Dict[str, Any]]:
    """Apriori + 連帶規則（帶診斷）。"""
    try:
        KMeans, TransactionEncoder, apriori, association_rules = _lazy_import_ml()
    except Exception:
        # 套件缺少時直接回診斷
        return [], {"ok": False, "baskets_total": 0, "baskets_valid": 0, "reason": "algorithm_error"}

    # 組籃子鍵
    cust_key = df.get("customer_email", pd.Series([""] * len(df))).fillna("").str.strip()
    if not (cust_key.str.len() > 0).any():
        cust_key = df.get("customer_name", pd.Series([""] * len(df))).fillna("").str.strip()

    if df.get("order_id", pd.Series([None] * len(df))).notna().any():
        df["basket_id"] = df["order_id"].fillna("NA").astype(str)
    else:
        df["basket_id"] = cust_key + "_" + df["date"].dt.strftime("%Y-%m-%d")

    tx_series = df.groupby("basket_id")["product_name"].apply(lambda s: sorted(set(s)))
    tx_all = int(len(tx_series))
    tx = [t for t in tx_series.tolist() if len(t) >= 2]

    assoc_rules_out = []
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
                        assoc_rules_out.append({
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

    diag = {
        "ok": len(assoc_rules_out) > 0,
        "baskets_total": tx_all,
        "baskets_valid": int(len(tx)),
        "reason": reason,
    }
    return assoc_rules_out, diag

def _rfm(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float], Dict[str, list], Dict[str, Any]]:
    """RFM 分群（優先 KMeans；fallback 改用 R/F/M 加權分數 + 25/75 分位），並輸出加溫名單。"""
    try:
        KMeans, *_ = _lazy_import_ml()
    except Exception:
        KMeans = None

    # ====== 建客戶鍵 / 訂單鍵 ======
    cust_key = df.get("customer_email", pd.Series([""] * len(df))).fillna("").str.strip()
    if not (cust_key.str.len() > 0).any():
        cust_key = df.get("customer_name", pd.Series([""] * len(df))).fillna("").str.strip()
    df = df.join(cust_key.rename("customer_id"))

    if df.get("order_id", pd.Series([None] * len(df))).notna().any():
        df["basket_id"] = df["order_id"].fillna("NA").astype(str)
    else:
        df["basket_id"] = df["customer_id"].fillna("") + "_" + df["date"].dt.strftime("%Y-%m-%d")
    df["order_key"] = df["basket_id"]

    # ====== 客戶彙總 ======
    cust = (df.groupby("customer_id")
              .agg(last_date=("date", "max"),
                   orders=("order_key", pd.Series.nunique),
                   monetary=("revenue", "sum"))
              .reset_index())
    cust = cust[cust["customer_id"].fillna("").str.len() > 0]

    customers_total = int(cust["customer_id"].nunique())
    customers_used = int(len(cust))

    rfm_share = {"low": 0.0, "mid": 0.0, "high": 0.0}
    rfm_counts = {"low": 0, "mid": 0, "high": 0}
    rfm_avg_monetary = {"low": 0.0, "mid": 0.0, "high": 0.0}
    nurture: Dict[str, list] = {"mid": [], "low": []}
    reason = None
    method = None  # 新增：告知這次用 kmeans 或 quantile

    if len(cust) >= 3:
        max_date = df["date"].max()
        cust["recency"] = (max_date - cust["last_date"]).dt.days.astype(float).clip(lower=0)

        # ====== 先試 KMeans ======
        use_quantile = False
        if KMeans is not None:
            try:
                # 特徵標準化
                X = cust[["recency", "orders", "monetary"]].astype(float)
                X = (X - X.mean()) / (X.std(ddof=0) + 1e-9)

                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                cust["segment"] = km.fit_predict(X)

                counts = cust["segment"].value_counts()
                if (len(counts) < 3) or (counts.min() <= 1):
                    use_quantile = True
                else:
                    # 依各群平均 monetary 排序 → 指派 low/mid/high
                    rank = (cust.groupby("segment")["monetary"]
                              .mean().sort_values().reset_index())
                    rank["level"] = ["low", "mid", "high"]
                    m = dict(zip(rank["segment"], rank["level"]))
                    cust["level"] = cust["segment"].map(m)
                    method = "kmeans"
            except Exception:
                use_quantile = True
        else:
            use_quantile = True

        # ====== 退回：用 R/F/M 加權分數 + 25/75 分位（避免硬三等份，且考慮 R/F/M） ======
        if use_quantile:
            # rank 到 0~1：Recency 越小越好 → 用負號反向
            def to_rank(series: pd.Series, reverse: bool = False) -> pd.Series:
                s = series.astype(float)
                if reverse: s = -s
                order = s.rank(method="average", pct=True)  # 0~1
                return order

            r_rank = to_rank(cust["recency"], reverse=True)   # recency 越新越高分
            f_rank = to_rank(cust["orders"],  reverse=False)
            m_rank = to_rank(cust["monetary"], reverse=False)

            score = 0.2 * r_rank + 0.3 * f_rank + 0.5 * m_rank
            cust["rfm_score"] = score

            t1 = score.quantile(0.25)  # 25%
            t2 = score.quantile(0.75)  # 75%

            def lvl_sc(s):
                if s <= t1: return "low"
                elif s <= t2: return "mid"
                else: return "high"

            cust["level"] = cust["rfm_score"].apply(lvl_sc)
            method = "quantile"

        # ====== 統計輸出 ======
        levels = ["low", "mid", "high"]

        cnt_map = cust["level"].value_counts().to_dict()
        rfm_counts = {lv: int(cnt_map.get(lv, 0)) for lv in levels}

        avg_map = {lv: float(cust.loc[cust["level"] == lv, "monetary"].mean() or 0.0) for lv in levels}
        rfm_avg_monetary = {lv: round(avg_map.get(lv, 0.0), 2) for lv in levels}

        total_rev = float(cust["monetary"].sum() or 0.0)
        share_sum_map = {lv: float(cust.loc[cust["level"] == lv, "monetary"].sum()) for lv in levels}
        rfm_share = {lv: (round(share_sum_map.get(lv, 0.0) / total_rev, 4) if total_rev > 0 else 0.0) for lv in levels}

        # ====== 加溫名單（mid/low） ======
        r_th = 45
        m_th_mid = cust.loc[cust["level"] == "mid", "monetary"].median() if (cust["level"] == "mid").any() else 0.0
        m_th_low = cust.loc[cust["level"] == "low", "monetary"].median() if (cust["level"] == "low").any() else 0.0

        def build_list(df_seg, m_th):
            seg = df_seg[(df_seg["recency"] > r_th) & (df_seg["monetary"] >= m_th)]
            out = (seg.sort_values("monetary", ascending=False)
                    .head(200)[["customer_id", "last_date", "orders", "monetary"]]
                    .rename(columns={
                        "customer_id": "customer_email",
                        "last_date": "last_order_date",
                        "orders": "lifetime_orders",
                        "monetary": "lifetime_revenue"
                    })
                    .to_dict(orient="records"))
            return out

        nurture["mid"] = build_list(cust[cust["level"] == "mid"], m_th_mid)
        nurture["low"] = build_list(cust[cust["level"] == "low"], m_th_low)

    else:
        reason = "no_customer_key_or_too_few"
        method = "na"

    diag = {
        "ok": customers_used >= 3,
        "customers_total": customers_total,
        "customers_used": customers_used,
        "reason": reason,
        "method": method,  # 新增：kmeans / quantile / na
    }
    return rfm_share, rfm_counts, rfm_avg_monetary, nurture, diag


def _potential_products(df: pd.DataFrame,
                        top5_names: set,
                        monthly_ordered: pd.DataFrame) -> list:
    """最後三個月營收加總、最近月 MoM > 20% 的上升產品，排除 Top5。"""
    out = []
    df["yyyymm"] = df["date"].dt.to_period("M").astype(str)
    prod_m = (df.groupby(["product_name", "yyyymm"], as_index=False)["revenue"].sum()
                .sort_values(["product_name", "yyyymm"]))
    if not prod_m.empty and len(monthly_ordered) >= 3:
        last3 = sorted(monthly_ordered["yyyymm"].unique())[-3:]
        recent = prod_m[prod_m["yyyymm"].isin(last3)]
        if not recent.empty:
            g = recent.pivot(index="product_name", columns="yyyymm", values="revenue").fillna(0.0)
            if g.shape[1] == 3:
                # 最後一個月相對倒數第二個月
                g["mom_last"] = (g.iloc[:, 2] - g.iloc[:, 1]) / (g.iloc[:, 1].replace(0, np.nan))
                g["sum3"] = g.sum(axis=1)
                cand = g[(g["mom_last"] > 0.2) & (~g.index.isin(top5_names))]
                cand = cand.sort_values(["mom_last", "sum3"], ascending=False).head(5)
                for name, row in cand.iterrows():
                    out.append({
                        "product_name": name,
                        "mom_last": round(float(row["mom_last"]), 3),
                        "sum3_revenue": round(float(row["sum3"]), 2)
                    })
    return out

# ========================= API ================================
@app.post("/assoc_window")
def assoc_window(payload: AnalyzeAssocIn, request: Request):
    n = len(payload.rows or [])
    if n == 0:
        return {"error": "no_rows",
                "diagnostics": {"assoc": {"ok": False, "reason": "empty_payload"}}}

    df = pd.DataFrame([r.dict() for r in payload.rows])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df = df.dropna(subset=["date", "product_name"])

    try:
        s = pd.to_datetime(payload.date_from)
        e = pd.to_datetime(payload.date_to)
    except Exception:
        raise HTTPException(status_code=422, detail={"error": "invalid_dates"})

    win_df = df[(df["date"] >= s) & (df["date"] <= e)].copy()
    if win_df.empty:
        return {
            "assoc_rules": [],
            "diagnostics": {
                "assoc": {"ok": False, "reason": "window_empty", "baskets_total": 0, "baskets_valid": 0},
                "window": {"from": payload.date_from, "to": payload.date_to, "rows": 0}
            }
        }

    assoc_rules, assoc_diag = _assoc_rules(win_df)
    return {
        "assoc_rules": assoc_rules,
        "diagnostics": {
            "assoc": assoc_diag,
            "window": {"from": payload.date_from, "to": payload.date_to, "rows": int(len(win_df))}
        }
    }

@app.post("/analyze")
async def analyze(payload: AnalyzeIn, request: Request):
    rid = getattr(request.state, "rid", "-")

    # Guard: row count
    n = len(payload.rows or [])
    if n == 0:
        return {"error": "no_rows",
                "diagnostics": {"required": {"ok": False, "reason": "empty_payload"}}}
    if n > MAX_ROWS:
        raise HTTPException(status_code=422, detail={"error": "too_many_rows", "max_rows": MAX_ROWS, "got": n})

    # Build DataFrame + clean
    df = pd.DataFrame([r.dict() for r in payload.rows])

    # Required fields
    required_cols = ["date", "product_name", "revenue"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {"error": "missing_required_fields",
                "diagnostics": {"required": {"ok": False, "reason": "missing_fields", "missing": missing}}}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", "product_name"])
    if df.empty:
        return {"error": "invalid_required_values",
                "diagnostics": {"required": {"ok": False, "reason": "all_rows_invalid"}}}

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

    # Assoc + RFM with timeouts（並行）
    assoc_rules, assoc_diag = [], {"ok": False, "reason": "timeout", "baskets_total": 0, "baskets_valid": 0}
    rfm_share = {"low": 0.0, "mid": 0.0, "high": 0.0}
    rfm_counts = {"low": 0, "mid": 0, "high": 0}
    rfm_avg_monetary = {"low": 0.0, "mid": 0.0, "high": 0.0}
    nurture_mid: list = []
    nurture_low: list = []
    rfm_diag = {"ok": False, "reason": "timeout", "customers_total": 0, "customers_used": 0}

    with futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_assoc = ex.submit(_assoc_rules, df.copy())
        f_rfm   = ex.submit(_rfm, df.copy())

        try:
            assoc_rules, assoc_diag = f_assoc.result(timeout=ALGO_TIMEOUT_SEC)
        except futures.TimeoutError:
            assoc_diag = {"ok": False, "reason": "timeout", "baskets_total": 0, "baskets_valid": 0}
            logging.warning(f"[{rid}] assoc timeout")

        try:
            rfm_share, rfm_counts, rfm_avg_monetary, nurture_dict, rfm_diag = f_rfm.result(timeout=ALGO_TIMEOUT_SEC)
            nurture_mid = nurture_dict.get("mid", [])
            nurture_low = nurture_dict.get("low", [])
        except futures.TimeoutError:
            rfm_diag = {"ok": False, "reason": "timeout", "customers_total": 0, "customers_used": 0}
            logging.warning(f"[{rid}] rfm timeout")

    potential_products = _potential_products(df.copy(), set(top5_df["product_name"].tolist()), monthly_df)

    return {
        "top5_products": top5,
        "monthly": monthly_df.to_dict(orient="records"),
        "mom": mom,
        "wow": wow,
        "assoc_rules": assoc_rules,
        "rfm_share": rfm_share,
        "rfm_counts": rfm_counts,
        "rfm_avg_monetary": rfm_avg_monetary,
        "nurture_list": (nurture_mid + nurture_low),  # 舊前端相容
        "nurture_mid": nurture_mid,
        "nurture_low": nurture_low,
        "potential_products": potential_products,
        "diagnostics": {"assoc": assoc_diag, "rfm": rfm_diag, "required": {"ok": True}},
    }


# 可本機測試時啟動（Render/Cloud Run 不需要）
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
