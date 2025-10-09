# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import streamlit.components.v1 as components  # for embedding Power BI

st.set_page_config(page_title="Hotel AI Dashboard", layout="wide")

# -------------------------------------
# Cached Helpers
# -------------------------------------
@st.cache_data
def load_data(uploaded_file):
    """Load uploaded data with caching."""
    if uploaded_file.name.endswith("xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

@st.cache_resource
def train_kmeans(data, n_clusters=3):
    """Train KMeans clustering (cached)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return scaler, kmeans, clusters

@st.cache_resource
def train_rf(X_train, y_train):
    """Train Random Forest classifier (cached)."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def fit_forecast(ts):
    """Fit Holt-Winters model and return forecast (cached)."""
    model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=30)
    fit = model.fit()
    forecast = fit.forecast(30)
    return fit, forecast


# -------------------------------------
# Load Data
# -------------------------------------
st.sidebar.header("Upload Booking Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("Please upload your bookings Excel/CSV file.")
    st.stop()

# -------------------------------------
# Sidebar Navigation
# -------------------------------------
menu = st.sidebar.radio(
    "Navigate",
    ["📘 Introduction", "🔮 Upsell Prediction", "❌ Cancellation Prediction", "📈 Revenue Forecasting", "📊 Power BI Dashboard"]
)

# -------------------------------------
# Introduction Page (Enhanced)
# -------------------------------------
if menu == "📘 Introduction":
    st.title("🏨 Hotel AI Dashboard")
    st.markdown("### 🤖 Smart Insights for Hotel Revenue Optimization")

    st.markdown("""
    Welcome to the **Hotel AI Dashboard** — your one-stop solution for data-driven decision-making 
    in hotel management.  
    This interactive dashboard uses **Machine Learning** to analyze booking, guest, and revenue data 
    to help you make smarter, faster decisions.

    ---
    ### 🚀 Key Features
    - 🔮 **Upsell Prediction** → Identify guests most likely to accept premium offers  
    - ❌ **Cancellation Prediction** → Detect high-risk bookings before they cancel  
    - 📈 **Revenue Forecasting** → Predict next 30 days’ revenue with confidence intervals  
    - 📊 **Power BI Integration** → Explore interactive visual insights  
    ---
    """)

    st.markdown("### 🎯 Project Objectives")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 📅 Maximize **Occupancy Rate**  
        - 💰 Optimize **Average Daily Rate (ADR)**  
        """)
    with col2:
        st.markdown("""
        - 📊 Improve **Revenue Forecasting Accuracy**  
        - 🔍 Enhance **Operational Decision-Making**  
        """)

    st.markdown("---")
    st.markdown("### 📂 Data Requirements")
    st.info("""
    Upload your **Bookings dataset (CSV or Excel)** to start exploring insights.  
    The dataset should ideally include:
    - `checkin_date`, `checkout_date`
    - `lead_time`, `ADR`, `nights`
    - `booking_status`, `revenue`, and guest details
    """)

    st.markdown("---")
    st.markdown("### 🔎 Quick Data Preview")
    if df is not None and not df.empty:
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.warning("Please upload a dataset to see the preview.")

# -------------------------------------
# Upsell Prediction (Upgraded + Fixed)
# -------------------------------------
elif menu == "🔮 Upsell Prediction":
    st.header("🔮 Upsell Prediction – Enhanced Guest Segmentation")

    # Required features for clustering
    upsell_features = ["lead_time", "ADR", "revenue", "nights"]
    df_clean = df[upsell_features].dropna()

    if not df_clean.empty:
        # Train model
        scaler, kmeans, clusters = train_kmeans(df_clean)

        # Assign cluster labels
        df.loc[df_clean.index, "upsell_cluster"] = clusters

        # Calculate average revenue per cluster and rank them
        cluster_revenue = (
            df.loc[df_clean.index]
            .groupby("upsell_cluster")["revenue"]
            .mean()
            .sort_values()
        )
        revenue_rank = {cid: i for i, cid in enumerate(cluster_revenue.index)}

        # Map revenue-based cluster ranking
        df.loc[df_clean.index, "upsell_score"] = df.loc[df_clean.index, "upsell_cluster"].map(revenue_rank)

        # Dynamic labeling based on number of unique clusters
        unique_scores = sorted(df["upsell_score"].dropna().unique())
        score_labels = {}

        if len(unique_scores) == 2:
            score_labels = {
                unique_scores[0]: "Low Potential",
                unique_scores[1]: "High Potential",
            }
        elif len(unique_scores) >= 3:
            score_labels = {
                unique_scores[0]: "Low Potential",
                unique_scores[1]: "Medium Potential",
                unique_scores[-1]: "High Potential",
            }

        # Final upsell label with fallback
        df["upsell_label"] = df["upsell_score"].map(score_labels).fillna("Unknown")

        # -----------------------------
        # 🎨 Visualization
        # -----------------------------
        st.subheader("Guest Segmentation Visualization")

        import plotly.express as px

        fig = px.scatter(
            df_clean,
            x="ADR",
            y="revenue",
            color=df.loc[df_clean.index, "upsell_label"],
            hover_data=["lead_time", "nights"],
            title="Upsell Segmentation: ADR vs Revenue",
            labels={"ADR": "Average Daily Rate", "revenue": "Total Revenue"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # 📊 Cluster Distribution
        # -----------------------------
        st.subheader("Upsell Potential Distribution")
        st.bar_chart(df["upsell_label"].value_counts())

        # -----------------------------
        # 💰 Top Upsell Candidates
        # -----------------------------
        st.subheader("Top 10 Guests for Upselling")
        if "customer_id" in df.columns:
            top_upsell = (
                df[df["upsell_label"].isin(["High Potential", "Medium Potential"])]
                .sort_values("revenue", ascending=False)
                .head(10)
            )
            st.dataframe(
                top_upsell[
                    ["customer_id", "ADR", "revenue", "nights", "upsell_label"]
                ]
            )
        else:
            st.info("No 'customer_id' column found – showing top revenue guests instead.")
            st.dataframe(
                df.sort_values("revenue", ascending=False)[
                    ["ADR", "revenue", "nights", "upsell_label"]
                ].head(10)
            )

        # -----------------------------
        # 🧠 Insight Summary
        # -----------------------------
        st.markdown("""
        ### Insights:
        - **High Potential Guests**: Most profitable customers – ideal for premium offers or upgrades.  
        - **Medium Potential Guests**: May respond well to small-value add-ons (e.g., breakfast deals).  
        - **Low Potential Guests**: Budget-conscious or short-stay visitors – minimal upsell opportunity.
        """)
    else:
        st.info("Not enough valid data for upsell prediction. Please check if 'lead_time', 'ADR', 'revenue', and 'nights' columns exist.")


# -------------------------------------
# Cancellation Prediction (Optimized for Speed + Clarity)
# -------------------------------------
elif menu == "❌ Cancellation Prediction":
    st.header("❌ Cancellation Prediction – Smart Booking Risk Model")

    if "booking_status" in df.columns:
        # Create flag
        df["cancel_flag"] = df["booking_status"].apply(
            lambda x: 1 if str(x).lower() in ["canceled", "cancelled", "no-show"] else 0
        )

        # Slightly richer feature set
        features = ["lead_time", "ADR", "nights"]
        df_model = df.dropna(subset=features + ["cancel_flag"])

        if not df_model.empty:
            X = df_model[features]
            y = df_model["cancel_flag"]

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Cached lightweight model training
            @st.cache_resource
            def train_fast_rf(X_train, y_train):
                model = RandomForestClassifier(
                    random_state=42,
                    n_estimators=150,
                    max_depth=8,
                    class_weight="balanced"
                )
                model.fit(X_train, y_train)
                return model

            model = train_fast_rf(X_train, y_train)

            # Evaluate model
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Model Accuracy: {acc:.2f}")

            # Classification report
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds))

            # 🔍 Feature importance
            importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.write("### 🔍 Feature Importance")
            st.bar_chart(importances)

            # ✨ Visualization: Cancellation probability by lead time
            import plotly.express as px
            df_model["predicted_prob"] = model.predict_proba(X)[:, 1]
            fig = px.scatter(
                df_model,
                x="lead_time",
                y="predicted_prob",
                color=df_model["cancel_flag"].map({1: "Canceled", 0: "Not Canceled"}),
                title="Cancellation Probability vs Lead Time",
                labels={"lead_time": "Lead Time (Days)", "predicted_prob": "Predicted Cancellation Probability"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Live form for prediction
            st.subheader("🧾 Predict Cancellation for New Booking")
            lead_time = st.number_input("Lead Time (days)", 0, 365, 10)
            adr = st.number_input("ADR (Average Daily Rate)", 0.0, 1000.0, 120.0)
            nights = st.number_input("Number of Nights", 1, 30, 2)

            new_pred = model.predict_proba([[lead_time, adr, nights]])[0][1]
            st.write(f"🔮 Predicted Cancellation Probability: **{new_pred * 100:.1f}%**")

            # Interpretation
            if new_pred > 0.7:
                st.error("⚠️ High Risk: Recommend prepayment or reminder.")
            elif new_pred > 0.4:
                st.warning("⚠️ Medium Risk: Monitor and confirm booking.")
            else:
                st.success("✅ Low Risk: Booking likely to stay confirmed.")

        else:
            st.info("Not enough valid rows for cancellation prediction.")
    else:
        st.info("No 'booking_status' column found in dataset.")

# -------------------------------------
# Revenue Forecasting (Optimized)
# -------------------------------------
elif menu == "📈 Revenue Forecasting":
    st.header("📈 Revenue Forecasting – Smarter Revenue Trend Analysis")

    if "checkin_date" in df.columns and "revenue" in df.columns:
        # Prepare time series
        df["checkin_date"] = pd.to_datetime(df["checkin_date"], errors="coerce")
        ts = df.groupby("checkin_date")["revenue"].sum().asfreq("D").fillna(0)

        if not ts.empty:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            @st.cache_resource
            def fit_fast_forecast(ts):
                """Lightweight cached Holt-Winters model"""
                model = ExponentialSmoothing(
                    ts, trend="add", seasonal="add", seasonal_periods=7
                ).fit()
                forecast = model.forecast(30)
                return model, forecast

            model, forecast = fit_fast_forecast(ts)

            # Combine for plotting
            forecast_df = pd.DataFrame({
                "date": forecast.index,
                "forecast": forecast.values
            })

            # 🔢 KPIs
            avg_revenue = ts[-30:].mean()
            growth_rate = ((forecast.mean() - avg_revenue) / avg_revenue) * 100 if avg_revenue != 0 else 0
            peak_forecast = forecast.max()

            col1, col2, col3 = st.columns(3)
            col1.metric("📅 Avg. Revenue (Last 30 Days)", f"₹{avg_revenue:,.0f}")
            col2.metric("📈 Projected Growth (Next 30d)", f"{growth_rate:.1f}%")
            col3.metric("💰 Peak Forecast Revenue", f"₹{peak_forecast:,.0f}")

            # 📊 Interactive Plot
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts.values, mode="lines", name="Historical", line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["date"], y=forecast_df["forecast"],
                mode="lines+markers", name="Forecast (Next 30d)", line=dict(dash="dash", width=3)
            ))
            fig.update_layout(
                title="📈 30-Day Revenue Forecast",
                xaxis_title="Date",
                yaxis_title="Revenue (₹)",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 🔍 Show data table (optional)
            with st.expander("View Forecast Data"):
                st.dataframe(forecast_df.style.format({"forecast": "{:,.0f}"}))

        else:
            st.info("No valid revenue data available for forecasting.")
    else:
        st.info("Need 'checkin_date' and 'revenue' columns for forecasting.")

# -------------------------------------
# Power BI Dashboard
# -------------------------------------
elif menu == "📊 Power BI Dashboard":
    st.header("📊 Power BI Dashboard")
    st.markdown("This section shows an embedded Power BI dashboard for deeper insights.")

    # Replace with your Power BI embed link
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=6a53bf14-0d57-4142-9215-87bec7b1364e&autoAuth=true&ctid=ea9a695f-d336-4931-b395-0191eea5c2e0"

    components.iframe(power_bi_url, height=600, scrolling=True)
