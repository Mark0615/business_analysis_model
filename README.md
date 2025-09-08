# E-Commerce Product Business Analytics Model

上傳銷售 CSV，一鍵得到趨勢、客群（RFM × KMeans）、商品分析與關聯規則洞察。  
前端採 **Next.js**（Cloudflare Pages），後端採 **FastAPI**（Render）。

- **Demo（Frontend）**: <https://business-analysis-model.pages.dev/>  ← 換成你的正式網址
- **API（Backend）**: <https://business-analysis-model-backend.onrender.com/>  ← 只做參考，不需直接開放給使用者

---

## ✨ Features

- CSV 上傳（自動辨識常見欄位）
- Data Summary：總營收、訂單數、客戶數、客單價、月營收趨勢
- Customer Analysis：RFM 計算、KMeans 分群、各群消費頻次與平均客單
- Product Analysis：暢銷 Top/折扣占比、地區分析（依你資料而定）
- （可選）關聯規則分析與診斷訊息
- 完整 CORS、環境變數化 API 位址；支援本機/雲端一致配置

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
├─ backend/
│ ├─ main.py # FastAPI App (含 /analyze, /health)
│ ├─ requirements.txt
│ └─ ... # 分析/模型相關模組
├─ frontend/
│ ├─ src/app/page.tsx # 主頁 UI 與分析觸發
│ ├─ next.config.ts
│ ├─ package.json
│ └─ ...
├─ data_set/ # 本機資料（.gitignore 排除）
├─ .gitignore
└─ README.md
