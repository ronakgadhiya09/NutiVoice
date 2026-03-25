import hashlib
import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import data_store
import insights
import nutrition_service
from data_store import init_schema

load_dotenv()
init_schema()

st.set_page_config(page_title="NutriVoice", page_icon="🎙️", layout="wide")

st.markdown(
    """
<style>
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(20,20,30,0.8), rgba(40,40,55,0.8));
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    transition: transform 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
}
.stProgress .st-bo {
    background-color: #00d2ff;
}
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #00d2ff, #3a7bd5);
}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_GOALS = {"Calories": 2000, "Protein": 150, "Carbs": 250, "Fats": 70}

if "goals" not in st.session_state:
    st.session_state.goals = data_store.load_goals(DEFAULT_GOALS)
if "voice_clip_hash" not in st.session_state:
    st.session_state.voice_clip_hash = None
if "voice_draft" not in st.session_state:
    st.session_state.voice_draft = ""
if "date_filter_mode" not in st.session_state:
    st.session_state.date_filter_mode = "Today"
if st.session_state.pop("_clear_voice_draft_pending", False):
    st.session_state.voice_draft = ""
    st.session_state.voice_clip_hash = None

api_key = os.getenv("GROQ_API_KEY")

# -- Sidebar: Config --
with st.sidebar:
    st.title("Settings")
    st.subheader("Daily goals")
    st.session_state.goals["Calories"] = int(
        st.number_input(
            "Target calories (kcal)",
            value=int(st.session_state.goals["Calories"]),
            step=100,
        )
    )
    st.session_state.goals["Protein"] = int(
        st.number_input(
            "Target protein (g)",
            value=int(st.session_state.goals["Protein"]),
            step=10,
        )
    )
    st.session_state.goals["Carbs"] = int(
        st.number_input(
            "Target carbs (g)",
            value=int(st.session_state.goals["Carbs"]),
            step=10,
        )
    )
    st.session_state.goals["Fats"] = int(
        st.number_input(
            "Target fats (g)",
            value=int(st.session_state.goals["Fats"]),
            step=5,
        )
    )
    data_store.save_goals(st.session_state.goals)

st.title("NutriVoice")
st.markdown(
    f"**Voice-first nutrition logging** · Local history · Hybrid lookup (Open Food Facts + AI) · "
    f"Today: **{date.today().strftime('%B %d, %Y')}**"
)

# -- Date range for dashboard --
mode_options = ["Today", "Last 7 days", "Last 30 days", "Custom range"]
mode = st.selectbox(
    "Dashboard period",
    mode_options,
    index=mode_options.index(st.session_state.date_filter_mode)
    if st.session_state.date_filter_mode in mode_options
    else 0,
    key="dashboard_period",
)
st.session_state.date_filter_mode = mode

today = date.today()
if mode == "Today":
    range_start, range_end = today, today
elif mode == "Last 7 days":
    range_start, range_end = today - timedelta(days=6), today
elif mode == "Last 30 days":
    range_start, range_end = today - timedelta(days=29), today
else:
    c_s, c_e = st.columns(2)
    with c_s:
        range_start = st.date_input("Start", value=today - timedelta(days=6), key="cust_start")
    with c_e:
        range_end = st.date_input("End", value=today, key="cust_end")
    if range_start > range_end:
        st.error("Start date must be on or before end date.")
        st.stop()

rows = data_store.fetch_items_in_range(range_start, range_end)
period_days = (range_end - range_start).days + 1

# Trends: last 30 days of data for charts (fixed window)
trend_start, trend_end = today - timedelta(days=29), today
rows_trend = data_store.fetch_items_in_range(trend_start, trend_end)
df_trend = insights.daily_totals_dataframe(rows_trend)

# Last 7 days for insights rules
insight_start, insight_end = today - timedelta(days=6), today
rows_insight = data_store.fetch_items_in_range(insight_start, insight_end)
rows_today_only = data_store.fetch_items_in_range(today, today)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Log your meal")
    st.caption("Prefer voice: record, review the transcript, then parse and save.")

    tab_voice, tab_text = st.tabs(["Voice (primary)", "Text"])

    with tab_voice:
        audio_value = st.audio_input("Record what you ate")
        if audio_value:
            audio_bytes = audio_value.getvalue()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            if audio_hash != st.session_state.voice_clip_hash:
                with st.spinner("Transcribing with device speech recognition…"):
                    transcript, err = nutrition_service.transcribe_audio_only(
                        audio_bytes, audio_value.type or ""
                    )
                st.session_state.voice_clip_hash = audio_hash
                st.session_state.voice_draft = transcript or ""
                if err:
                    st.warning(err)
                elif transcript:
                    st.success("Transcript ready — edit below if needed, then log.")
                if transcript or err:
                    st.rerun()

        st.text_area(
            "Transcript (edit if needed)",
            height=120,
            key="voice_draft",
        )
        if st.button("Parse & log meal", type="primary", key="log_voice"):
            if not api_key:
                st.error("Set GROQ_API_KEY in your .env file.")
            elif not (st.session_state.get("voice_draft") or "").strip():
                st.error("No transcript to parse. Record audio or type in the box.")
            else:
                draft = st.session_state.voice_draft.strip()
                with st.spinner("Extracting foods and nutrition…"):
                    parsed, meal_type = nutrition_service.resolve_meal_text(draft, api_key)
                if parsed:
                    data_store.add_meal_with_items(
                        meal_type=meal_type,
                        source="voice",
                        raw_input=None,
                        transcript=draft,
                        items=parsed,
                    )
                    st.success(f"Logged {len(parsed)} item(s).")
                    st.session_state._clear_voice_draft_pending = True
                    st.rerun()
                else:
                    st.error("Could not parse foods. Try a clearer description.")

    with tab_text:
        text_input = st.text_area(
            "What did you eat?",
            placeholder='e.g. "I had 2 eggs and a bowl of poha for breakfast"',
            height=100,
            key="meal_text",
        )
        if st.button("Log text", type="primary", key="log_text"):
            if not api_key:
                st.error("Set GROQ_API_KEY in your .env file.")
            elif not text_input.strip():
                st.error("Enter a meal description.")
            else:
                with st.spinner("Extracting foods and nutrition…"):
                    parsed, meal_type = nutrition_service.resolve_meal_text(
                        text_input.strip(), api_key
                    )
                if parsed:
                    data_store.add_meal_with_items(
                        meal_type=meal_type,
                        source="text",
                        raw_input=text_input.strip(),
                        transcript=None,
                        items=parsed,
                    )
                    st.success(f"Logged {len(parsed)} item(s).")
                    st.rerun()
                else:
                    st.error("Could not parse foods. Try being more specific.")

with col2:
    st.subheader("Dashboard")
    g = st.session_state.goals
    total_cals = sum(float(r.get("Calories", 0) or 0) for r in rows)
    total_protein = sum(float(r.get("Protein", 0) or 0) for r in rows)
    total_carbs = sum(float(r.get("Carbs", 0) or 0) for r in rows)
    total_fats = sum(float(r.get("Fats", 0) or 0) for r in rows)

    col_c, col_p, col_cb, col_f = st.columns(4)
    col_c.metric("Calories", f"{total_cals:.0f} / {g['Calories']}")
    col_p.metric("Protein (g)", f"{total_protein:.0f} / {g['Protein']}")
    col_cb.metric("Carbs (g)", f"{total_carbs:.0f} / {g['Carbs']}")
    col_f.metric("Fats (g)", f"{total_fats:.0f} / {g['Fats']}")

    denom = float(g["Calories"]) if g["Calories"] else 1.0
    st.progress(min(1.0, total_cals / denom), text="Calories vs goal (selected period)")

    target_scale = 1 if period_days <= 1 else period_days
    fig = go.Figure(
        data=[
            go.Bar(
                name="Consumed",
                x=["Protein", "Carbs", "Fats"],
                y=[total_protein, total_carbs, total_fats],
                marker_color="#00d2ff",
            ),
            go.Bar(
                name="Target × days in period" if period_days > 1 else "Target",
                x=["Protein", "Carbs", "Fats"],
                y=[
                    g["Protein"] * target_scale,
                    g["Carbs"] * target_scale,
                    g["Fats"] * target_scale,
                ],
                marker_color="#3a7bd5",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Macronutrients (selected period)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Trends (last 30 days)")
if df_trend.empty:
    st.info("No data in the last 30 days yet.")
else:
    g = st.session_state.goals
    fig2 = go.Figure()
    fig2.add_bar(
        x=df_trend["date"],
        y=df_trend["Calories"],
        name="Calories",
        marker_color="#00d2ff",
    )
    fig2.add_scatter(
        x=df_trend["date"],
        y=[g["Calories"]] * len(df_trend),
        mode="lines",
        name="Daily calorie goal",
        line=dict(color="#ff9f43", dash="dash"),
    )
    fig2.update_layout(
        title="Daily calories vs goal",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Date",
        yaxis_title="kcal",
    )
    st.plotly_chart(fig2, width="stretch")

    fig3 = go.Figure()
    fig3.add_scatter(
        x=df_trend["date"],
        y=df_trend["Protein"],
        mode="lines+markers",
        name="Protein (g)",
        line=dict(color="#26de81"),
    )
    fig3.add_scatter(
        x=df_trend["date"],
        y=[g["Protein"]] * len(df_trend),
        mode="lines",
        name="Protein goal",
        line=dict(color="#ff9f43", dash="dash"),
    )
    fig3.update_layout(
        title="Daily protein vs goal",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Date",
        yaxis_title="g",
    )
    st.plotly_chart(fig3, width="stretch")

st.divider()
st.subheader("Insights & suggestions")
in_cols = st.columns(2)
with in_cols[0]:
    st.markdown("**Last 7 days (when you logged)**")
    for line in insights.rule_based_insights(rows_insight, st.session_state.goals):
        st.markdown(f"- {line}")
with in_cols[1]:
    st.markdown("**Today**")
    for line in insights.suggestion_cards(rows_today_only, st.session_state.goals):
        st.markdown(f"- {line}")

st.divider()
st.subheader("Food log (selected period)")
if not rows:
    st.write("No items in this period.")
else:
    display_cols = [
        "created_at",
        "Name",
        "Calories",
        "Protein",
        "Carbs",
        "Fats",
        "nutrition_source",
        "confidence_status",
        "source",
    ]
    df_log = pd.DataFrame(rows)
    for c in display_cols:
        if c not in df_log.columns:
            df_log[c] = None
    st.dataframe(df_log[display_cols], use_container_width=True)

    low = [
        r
        for r in rows
        if r.get("confidence_status") in ("needs_review", "fallback")
        and r.get("nutrition_source") != "user_edited"
    ]
    if low:
        with st.expander("Adjust nutrition for low-confidence items"):
            edf = pd.DataFrame(
                [
                    {
                        "item_id": r["item_id"],
                        "Name": r["Name"],
                        "Calories": float(r["Calories"]),
                        "Protein": float(r["Protein"]),
                        "Carbs": float(r["Carbs"]),
                        "Fats": float(r["Fats"]),
                    }
                    for r in low
                ]
            )
            edited = st.data_editor(
                edf,
                disabled=["item_id", "Name"],
                use_container_width=True,
                key="nutrition_fix_editor",
            )
            if st.button("Save corrections", key="save_nutrition_fixes"):
                for _, er in edited.iterrows():
                    data_store.update_food_item_nutrition(
                        str(er["item_id"]),
                        calories=float(er["Calories"]),
                        protein=float(er["Protein"]),
                        carbs=float(er["Carbs"]),
                        fats=float(er["Fats"]),
                        confidence_status="matched",
                    )
                st.success("Updated.")
                st.rerun()

    if mode == "Today" and rows:
        if st.button("Clear today's log", help="Deletes all entries logged today on this device."):
            data_store.delete_entries_for_local_date(today)
            st.session_state._clear_voice_draft_pending = True
            st.rerun()
