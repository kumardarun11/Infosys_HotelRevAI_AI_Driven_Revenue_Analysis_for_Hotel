
# 🏨 AI Driven Revenue Analysis for Hotel

### 📊 Power BI Dashboard • 🌐 Streamlit AI App • 💬 Power BI Chatbot

> 📝 **This project was developed as part of the Infosys Internship Program.**

This project delivers a **complete hotel revenue analytics ecosystem** by combining the **visual storytelling power of Power BI**, the **predictive intelligence of Streamlit**, and the **conversational capabilities of a Power BI Chatbot**.

Managers and analysts can **explore dashboards**, **predict future trends**, and **ask questions in natural language** — all in one integrated solution.

---
<img width="984" height="684" alt="image" src="https://github.com/user-attachments/assets/5c263294-10df-4d77-9672-b9da3a3b1645" />


## 🚀 Key Components

| Component                     | Description                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 📊 **Power BI Dashboard**     | Interactive visualizations for revenue, occupancy, segmentation, forecasting, and pricing.                    |
| 🌐 **Streamlit AI Dashboard** | Machine learning–powered web app for upsell prediction, cancellation risk, and revenue forecasting.           |
| 💬 **Power BI Chatbot**       | Natural language interface that allows users to query hotel data directly, powered by Power BI Q&A / Copilot. |

---

## 🧭 Power BI Dashboard (Core)

The Power BI Dashboard (`Hotel_Analysis.pbix`) is the **heart** of this project.
It provides management teams with an end-to-end **revenue strategy cockpit**.

### 📌 Key Dashboards

* **Occupancy & Revenue Metrics** – ADR, RevPAR, occupancy trends, filters
* **Guest Segmentation** – Purpose, loyalty, demographics, channel mix
* **Forecasting & Cancellation** – Future occupancy, heatmaps, booking trends
* **Revenue Strategy** – Upselling potential, channel optimization, seasonal pricing
* **Dynamic Pricing Model** – LightGBM-driven pricing recommendations visualized

📐 **Data Model:**

* Fact Table → `fact_bookings`
* Dimension Tables → `dim_hotels`, `dim_rooms`, `dim_date` (Star Schema)

🟡 **To use:**

1. Open `Hotel_Analysis.pbix` in [Power BI Desktop](https://powerbi.microsoft.com/desktop/)
2. Load or refresh data
3. Interact with dashboards & slicers
4. (Optional) Publish to Power BI Service for online access

---

## 💬 Power BI Chatbot

The **Power BI Q&A / Copilot chatbot** lets users **ask questions in plain English** and get **real-time visual answers**.
Examples:

* “Show me occupancy trends for August 2023”
* “What is the average daily rate by room type?”
* “Which booking channel generated the most revenue last quarter?”

### 🧠 Features

* Built using **Power BI Q&A visual** and/or **Power BI Copilot** (if enabled)
* Supports **ad-hoc queries** without needing to manually explore visuals
* Automatically generates charts, tables, and metrics based on the question
* Accessible within the Power BI report or via **Power BI Service chatbot interface**

---

## 🌐 Streamlit AI Dashboard

The `app.py` file provides a **machine learning companion web app**, built with **Streamlit**.

### ✨ Features

| Feature                       | Description                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| 🔮 **Upsell Prediction**      | Segment guests into Low / Medium / High potential using K-Means clustering |
| ❌ **Cancellation Prediction** | Predict risky bookings using Random Forest                                 |
| 📈 **Revenue Forecasting**    | 30-day Holt-Winters forecast with KPIs                                     |
| 📊 **Power BI Embedding**     | View your hosted Power BI Dashboard inside Streamlit                       |
| 🧠 **Data Upload**            | Bring your own CSV/Excel booking data for instant insights                 |

---

## 🧰 Tech Stack

* **Power BI** → Data modeling, dashboards, chatbot (Q&A / Copilot)
* **Python + Streamlit** → Interactive ML web app
* **Pandas, scikit-learn, Statsmodels** → Data prep, clustering, forecasting
* **Plotly** → Dynamic charts
* **LightGBM** → Demand-based pricing recommendations (visualized in Power BI)

---

## 🧱 Project Structure

```
📁 Infosys_HotelRevAI_AI_Driven_Revenue_Analysis_for_Hotel
├── 📊 Hotel_Analysis.pbix        # Power BI Dashboard + Chatbot
├── 🌐 app.py                     # Streamlit AI Web App
├── 📄 Presentation.pdf           # Project Presentation
├── 📄 README.md
└── 📁 data/                      # (Optional) Sample or user datasets
```

---

## ⚡ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Infosys_HotelRevAI_AI_Driven_Revenue_Analysis_for_Hotel.git
cd Infosys_HotelRevAI_AI_Driven_Revenue_Analysis_for_Hotel
```

### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

*(If missing, generate it with `pip freeze > requirements.txt`)*

---

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

👉 Open the browser ([http://localhost:8501](http://localhost:8501))
👉 Upload your dataset
👉 Access AI predictions + embedded Power BI dashboard

---

### 4️⃣ Open the Power BI Dashboard

1. Launch **Power BI Desktop**
2. Open `Hotel_Analysis.pbix`
3. Explore dashboards, slicers, and Q&A chatbot
4. Publish to Power BI Service to enable **online chatbot querying** and embedding inside Streamlit

---

## 📝 License

Licensed under the **MIT License** — feel free to use, adapt, and extend with proper attribution.

---

## 🌟 Why This Project Stands Out

✅ **Power BI Dashboard** — visually rich, interactive, and strategic
✅ **Streamlit AI App** — predictive insights & forecasting
✅ **Power BI Chatbot** — natural language exploration of hotel data
✅ **End-to-End Solution** — from raw data to intelligent decision-making
