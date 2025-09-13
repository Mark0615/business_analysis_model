# E-Commerce Product Business Analytics Model 
A one-page analytics platform that turns a CSV into real-time insights, including trends, RFM segmentation, market-basket rules, campaign audiences, and emerging products, plus one-click LLM-generated slide decks (PPTX). Frontend emphasizes UX & data cleaning, backend provides a tunable, robust analytics API.
**Live Site** : https://business-analysis-model.pages.dev/

---

## ✨ Features

- **資料上傳與清洗，自動辨認常見欄位** | Data upload and cleansing with automated detection of common fields.
- **清楚的資料視覺化圖表**｜Clear and intuitive data visualizations for actionable insights.
- **快速掌握 RFM 分群名單，可下載潛在顧客名單** | Rapid generation of RFM-based customer segmentation lists, with export support for potential customer targeting.
- **透過Apriori演算法掌握關聯商品** | Association rule mining with the Apriori algorithm to identify related products.
- **潛力商品偵測** | Detection of potential high-performing products through data-driven analysis.
- **一鍵產出簡報** | One-click generation of presentation-ready reports.

---

## 🧱 Tech Stack

- **Frontend**: Next.js, React, TypeScript, CSS
- **Hosting**: Cloudflare Pages
- **Backend**: FastAPI, Uvicorn, Python
- **ML/DS**: pandas, numpy, scikit-learn（KMeans / RFM 等）
- **Backend Hosting**: Render

---

## 📂 Work Flow
User upload CSV → Data cleaning and visualization→ Call backend run analysis model → Return insights

---

## 📊 Screenshots
Page 1 - Summary :
<img width="1277" height="901" alt="image" src="https://github.com/user-attachments/assets/94afee19-d126-47c9-aaed-da8981b1fbd3" />

Page 2 - Customer Analysis :
<img width="1258" height="921" alt="image" src="https://github.com/user-attachments/assets/c681379d-ee4b-49e6-9554-42dc63ff9734" />


