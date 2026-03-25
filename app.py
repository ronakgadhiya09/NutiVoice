import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import json
import os
import utils
import hashlib
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Nutrient Tracker", page_icon="🍎", layout="wide")

st.markdown("""
<style>
/* Add a premium feel with custom metric styling and dynamic colors */
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
""", unsafe_allow_html=True)

# -- Session State Init --
if "food_log" not in st.session_state:
    st.session_state.food_log = []
if "goals" not in st.session_state:
    st.session_state.goals = {"Calories": 2000, "Protein": 150, "Carbs": 250, "Fats": 70}
if "processed_audio_hashes" not in st.session_state:
    st.session_state.processed_audio_hashes = set()

# -- Sidebar: Config --
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.subheader("🎯 Daily Goals")
    st.session_state.goals["Calories"] = st.number_input("Target Calories (kcal)", value=st.session_state.goals["Calories"], step=100)
    st.session_state.goals["Protein"] = st.number_input("Target Protein (g)", value=st.session_state.goals["Protein"], step=10)
    st.session_state.goals["Carbs"] = st.number_input("Target Carbs (g)", value=st.session_state.goals["Carbs"], step=10)
    st.session_state.goals["Fats"] = st.number_input("Target Fats (g)", value=st.session_state.goals["Fats"], step=5)

# Try to get API Key
api_key = os.getenv("GROQ_API_KEY")

st.title("🍎 Simple Nutrient Tracker")
st.markdown(f"**Today's Date:** {date.today().strftime('%B %d, %Y')} | Powered by **Groq**")

# -- Main UI Layout --
col1, col2 = st.columns([1, 1])

# Column 1: Input forms
with col1:
    st.subheader("📝 Log your meal")
    st.markdown("Use text or voice to log what you ate.")
    
    tab1, tab2 = st.tabs(["Text Input", "Voice Input"])
    
    with tab1:
        text_input = st.text_area("What did you eat?", placeholder="e.g. I ate 2 eggs and a bowl of poha")
        if st.button("Log Text", type="primary"):
            if not api_key:
                st.error("Please provide a GROQ_API_KEY in your .env file.")
            elif text_input:
                with st.spinner("Parsing food data via Groq..."):
                    parsed_items = utils.parse_food_text(text_input, api_key)
                    if parsed_items:
                        st.session_state.food_log.extend(parsed_items)
                        st.success(f"Added {len(parsed_items)} items!")
                        st.rerun()
                    else:
                        st.error("Failed to parse food items. Try being more specific.")
                
    with tab2:
        audio_value = st.audio_input("Record what you ate")
        if audio_value:
            audio_bytes = audio_value.getvalue()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            
            if audio_hash not in st.session_state.processed_audio_hashes:
                if not api_key:
                    st.error("Please provide a GROQ_API_KEY in your .env file.")
                else:
                    with st.spinner("Processing voice data via Groq (Whisper)..."):
                        parsed_items = utils.parse_food_audio(audio_bytes, audio_value.type, api_key)
                        st.session_state.processed_audio_hashes.add(audio_hash)
                        if parsed_items:
                            st.session_state.food_log.extend(parsed_items)
                            st.success(f"Added {len(parsed_items)} items!")
                            st.rerun()
                        else:
                            st.error("Failed to parse audio. Try speaking more clearly.")

# Column 2: Dashboard
with col2:
    st.subheader("📊 Dashboard")
    
    # Calculate totals with type checking to avoid AttributeError: 'str' object has no attribute 'get'
    total_cals = sum(item.get("Calories", 0) for item in st.session_state.food_log if isinstance(item, dict))
    total_protein = sum(item.get("Protein", 0) for item in st.session_state.food_log if isinstance(item, dict))
    total_carbs = sum(item.get("Carbs", 0) for item in st.session_state.food_log if isinstance(item, dict))
    total_fats = sum(item.get("Fats", 0) for item in st.session_state.food_log if isinstance(item, dict))
    
    col_c, col_p, col_cb, col_f = st.columns(4)
    col_c.metric("Calories", f"{total_cals} / {st.session_state.goals['Calories']}")
    col_p.metric("Protein (g)", f"{total_protein} / {st.session_state.goals['Protein']}")
    col_cb.metric("Carbs (g)", f"{total_carbs} / {st.session_state.goals['Carbs']}")
    col_f.metric("Fats (g)", f"{total_fats} / {st.session_state.goals['Fats']}")
    
    st.progress(min(1.0, total_cals / st.session_state.goals["Calories"] if st.session_state.goals["Calories"] else 0), text="Calories Progress")
    
    # Macro Bar Chart
    fig = go.Figure(data=[
        go.Bar(name='Consumed', x=['Protein', 'Carbs', 'Fats'], y=[total_protein, total_carbs, total_fats], marker_color='#00d2ff'),
        go.Bar(name='Target', x=['Protein', 'Carbs', 'Fats'], y=[st.session_state.goals['Protein'], st.session_state.goals['Carbs'], st.session_state.goals['Fats']], marker_color='#3a7bd5')
    ])
    fig.update_layout(barmode='group', title="Macronutrients Progress (g)",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color="white"))
    st.plotly_chart(fig, width="stretch")
    
    st.subheader("Today's Food Log")
    if not st.session_state.food_log:
        st.write("No food logged yet today. Start eating!")
    else:
        df = pd.DataFrame(st.session_state.food_log)
        st.dataframe(df, use_container_width=True)
        
        if st.button("🗑️ Clear Today's Log", help="Reset all entries for today."):
            st.session_state.food_log = []
            st.session_state.processed_audio_hashes = set()
            st.rerun()
