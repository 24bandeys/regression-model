import calendar
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Monthly Demand Forecast", page_icon="📦", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")


model = load_model()
FEATURES = list(model.feature_names_in_)  # exact columns/order used in training

# Stores come from the model itself: store_1 was dropped (drop_first=True),
# the rest are the 'store_id_*' columns.
stores = ["store_1"] + [c.replace("store_id_", "") for c in FEATURES if c.startswith("store_id_")]
stores = sorted(stores, key=lambda s: int(s.split("_")[1]))


def predict_daily(store, price, promo, weekday, month):
    row = {c: 0 for c in FEATURES}
    row.update({"price": price, "promo": promo, "weekday": weekday, "month": month})
    col = f"store_id_{store}"
    if col in row:
        row[col] = 1
    return float(model.predict(pd.DataFrame([row])[FEATURES])[0])


def forecast_month(store, year, month, price, promo_days):
    n_days = calendar.monthrange(year, month)[1]
    rows = []
    for d in range(1, n_days + 1):
        weekday = calendar.weekday(year, month, d)  # Monday=0 ... same as dataset
        base = predict_daily(store, price, 0, weekday, month)
        promo = predict_daily(store, price, 1, weekday, month)
        rows.append({"day": d, "weekday": weekday, "no_promo": base, "promo": promo})
    df = pd.DataFrame(rows)
    # Spread the chosen number of promo days evenly across the month
    df["promo_flag"] = 0
    if promo_days > 0:
        idx = np.linspace(0, n_days - 1, promo_days).round().astype(int)
        df.loc[idx, "promo_flag"] = 1
    df["units"] = np.where(df["promo_flag"] == 1, df["promo"], df["no_promo"])
    return df


st.title("📦 Monthly Product Demand Forecast")
st.caption("Random Forest regression model · predicts units demanded per store per month")

c1, c2 = st.columns(2)
with c1:
    store = st.selectbox("Store", stores)
    year = st.number_input("Year", min_value=2019, max_value=2035, value=2024, step=1)
    month = st.selectbox(
        "Month", list(range(1, 13)), format_func=lambda m: calendar.month_name[m], index=0
    )
with c2:
    price = st.number_input("Average product price", min_value=0.0, value=50.0, step=1.0)
    n_days = calendar.monthrange(int(year), int(month))[1]
    promo_days = st.slider("Number of promotion days in the month", 0, n_days, 0)

if st.button("Forecast demand", type="primary"):
    res = forecast_month(store, int(year), int(month), price, promo_days)
    total = int(round(res["units"].sum()))

    st.metric(f"Forecast units demanded — {store}, {calendar.month_name[month]} {int(year)}", f"{total:,} units")

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg per day", f"{res['units'].mean():.1f}")
    m2.metric("Peak day", f"{res['units'].max():.0f}")
    m3.metric("Days in month", n_days)

    st.subheader("Daily forecast")
    st.line_chart(res.set_index("day")["units"])
    with st.expander("Show daily table"):
        out = res[["day", "weekday", "promo_flag", "units"]].copy()
        out["units"] = out["units"].round(0).astype(int)
        st.dataframe(out, width="stretch", hide_index=True)

with st.expander("About this model"):
    st.write(
        "Trained on daily retail sales (2019–2023) using price, promotion flag, weekday, "
        "month and store. The model predicts daily units; the app adds up every day of the "
        "selected month to give the monthly total. Test-set performance: "
        "R² ≈ 0.917, MAE ≈ 3.4 units/day."
    )
