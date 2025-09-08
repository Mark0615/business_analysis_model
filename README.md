# E-Commerce Product Business Analytics Model 
Upload your sales CSV and instantly get insights on trends, customer segments (RFM × KMeans), product analysis, and association rules.
Frontend built with **Next.js** (Cloudflare Pages), backend built with **FastAPI** (Render).

- **Demo（Frontend）**: <https://business-analysis-model.pages.dev/>

---

## ✨ Features

- CSV 上傳（自動辨識常見欄位，data, product_name, revenue 為必填
  CSV upload (auto-detects common fields; date, product_name, revenue are required)
  
- Data Summary：總營收、訂單數、客戶數、客單價、月營收趨勢
  Data Summary: total revenue, number of orders, number of customers, average order value, monthly revenue trends
  
- Customer Analysis：RFM 計算、KMeans 分群、各群消費頻次與平均客單
  Customer Analysis: RFM calculation, KMeans clustering, consumption frequency and average order value by segment
  
- Product Analysis：暢銷 Top/折扣占比、地區分析（依你資料而定）
  Product Analysis: best-sellers & discount ratios, regional analysis (based on your dataset)
  
- 完整 CORS、環境變數化 API 位址；支援本機/雲端一致配置
  Full CORS support, API endpoint via environment variables; consistent configuration for both local and cloud environments

---

## 🧱 Tech Stack

- **Frontend**: Next.js 15, React, TypeScript, Tailwind CSS
- **Hosting**: Cloudflare Pages
- **Backend**: FastAPI, Uvicorn, Python (≥3.11)
- **ML/DS**: pandas, numpy, scikit-learn（KMeans / RFM 等）
- **Backend Hosting**: Render

---

## 📂 Project Structure
repo-root/
- backend/
-- main.py # FastAPI App (含 /analyze, /health)
-- requirements.txt
-- 分析/模型相關模組
─ frontend/
-- src/app/page.tsx # 主頁 UI 與分析觸發
-- next.config.ts
-- package.json
-- ...
- data_set/ # 本機資料（.gitignore 排除）
-- .gitignore
-- README.md
