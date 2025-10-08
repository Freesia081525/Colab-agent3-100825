import io
import json
import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Optional Gemini integration
USE_GEMINI_DEFAULT = True
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# Try to import google.generativeai if available
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# ---------------------------------------
# WOW Themes: palette + fonts + accents
# ---------------------------------------
THEMES = {
    "UK Royal": {
        "palette": ["#00247D", "#C8102E", "#FFD700", "#FFFFFF", "#000000"],
        "font": "Georgia",
        "accent": "#FFD700",
        "bg": "#F2F4F8"
    },
    "Azure Coast": {
        "palette": ["#007FFF", "#00BFFF", "#2E8B57", "#F0FFFF", "#0F3D57"],
        "font": "Verdana",
        "accent": "#00BFFF",
        "bg": "#EEF7FF"
    },
    "Deep sea": {
        "palette": ["#001F3F", "#003366", "#004C99", "#00A3E0", "#66D9EF"],
        "font": "Trebuchet MS",
        "accent": "#00A3E0",
        "bg": "#0B1F2A"
    },
    "Ferrari sports car": {
        "palette": ["#DC0000", "#FFEB00", "#1E1E1E", "#2C2C2C", "#FFFFFF"],
        "font": "Helvetica",
        "accent": "#DC0000",
        "bg": "#1E1E1E"
    },
    "Norwaygean": {
        "palette": ["#1B3A4B", "#EFEFEF", "#F2545B", "#C1DFF0", "#537780"],
        "font": "Calibri",
        "accent": "#F2545B",
        "bg": "#F5FAFF"
    },
    "Mozart": {
        "palette": ["#4A3F35", "#D4AF37", "#F5F5DC", "#6B4E3D", "#1C1C1C"],
        "font": "Times New Roman",
        "accent": "#D4AF37",
        "bg": "#F9F6EE"
    },
    "J.S.Bach": {
        "palette": ["#4B0082", "#DAA520", "#EFE6DD", "#2E2E3A", "#FFFFFF"],
        "font": "Times New Roman",
        "accent": "#DAA520",
        "bg": "#F7F2EA"
    },
}

CORAL = "#FF6F61"

# ---------------------------------------
# Streamlit page config
# ---------------------------------------
st.set_page_config(page_title="Agent-based Visualization (Streamlit + HF Spaces)", layout="wide")

# ---------------------------------------
# Theme selector + CSS
# ---------------------------------------
with st.sidebar:
    st.markdown("## Theme")
    theme_name = st.selectbox("Select WOW theme", list(THEMES.keys()), index=0)
theme = THEMES[theme_name]

def inject_theme_css(theme):
    palette = theme["palette"]
    font = theme["font"]
    accent = theme["accent"]
    bg = theme["bg"]
    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            background-color: {bg} !important;
            font-family: '{font}', sans-serif;
        }}
        .wow-title {{
            color: {palette[0]};
            border-left: 8px solid {accent};
            padding-left: 12px;
            font-weight: 700;
            letter-spacing: 0.4px;
        }}
        .wow-sub {{
            color: {palette[2]};
            font-weight: 600;
        }}
        .wow-box {{
            background: rgba(255,255,255,0.7);
            border: 1px solid {accent};
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        .wow-accent {{
            color: {accent};
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
inject_theme_css(theme)

# ---------------------------------------
# Helpers: load dataset from csv/json
# ---------------------------------------
def detect_format_and_load(text_or_bytes, filename=None):
    if text_or_bytes is None:
        return None, "No dataset provided."

    # File bytes
    if isinstance(text_or_bytes, bytes):
        if filename and filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(text_or_bytes))
                return df, f"Loaded CSV: {filename}"
            except Exception as e:
                return None, f"Error reading CSV: {e}"
        else:
            # Try JSON
            try:
                obj = json.load(io.BytesIO(text_or_bytes))
                df = pd.json_normalize(obj)
                return df, f"Loaded JSON: {filename or 'uploaded file'}"
            except Exception as e:
                return None, f"Error reading JSON: {e}"

    # Pasted text
    if isinstance(text_or_bytes, str):
        txt = text_or_bytes.strip()
        # JSON first
        try:
            obj = json.loads(txt)
            df = pd.json_normalize(obj)
            return df, "Loaded JSON from pasted text"
        except Exception:
            pass
        # CSV fallback
        try:
            df = pd.read_csv(io.StringIO(txt))
            return df, "Loaded CSV from pasted text"
        except Exception as e:
            return None, f"Error reading pasted text as CSV/JSON: {e}"

    return None, "Unsupported dataset input format."

# ---------------------------------------
# Helpers: parse agents.yaml
# ---------------------------------------
def parse_agents_yaml(text_or_bytes):
    try:
        if isinstance(text_or_bytes, bytes):
            data = yaml.safe_load(io.StringIO(text_or_bytes.decode("utf-8")))
        else:
            data = yaml.safe_load(text_or_bytes)
    except Exception as e:
        return [], f"Error parsing YAML: {e}"

    if not data:
        return [], "Empty YAML content."

    agents = data.get("agents") or data.get("Agents") or data.get("AGENTS")
    if not agents or not isinstance(agents, list):
        return [], "No agents list found in YAML."

    cleaned = []
    for a in agents:
        name = a.get("name") or a.get("id") or a.get("title")
        viz = a.get("visualization") or {}
        cleaned.append({
            "name": name or "Unnamed Agent",
            "description": a.get("description", ""),
            "visualization": {
                "type": viz.get("type"),
                "x": viz.get("x"),
                "y": viz.get("y"),
                "hue": viz.get("hue"),
                "aggregate": viz.get("aggregate"),
            }
        })
    return cleaned, f"Parsed {len(cleaned)} agents."

# ---------------------------------------
# Optional: Gemini suggest visualization
# ---------------------------------------
def suggest_viz_with_gemini(api_key, df, agent_context=""):
    if not api_key or not GENAI_AVAILABLE:
        return None, "Gemini unavailable or no API key."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        cols = list(df.columns)
        sample = df.head(5).to_dict(orient="records")
        prompt = (
            "You are a data visualization assistant. Given dataframe columns and a brief agent description, "
            "return concise JSON with keys: type (bar|pie|scatter|hist), x, y, hue (optional), aggregate (sum|count|mean|none). "
            "Prefer categorical x with numeric y for bar; pie for categorical distribution; scatter for two numeric columns; hist for single numeric.\n\n"
            f"Columns: {cols}\nSample rows: {json.dumps(sample)}\nAgent context: {agent_context}\n"
            "Return only the JSON (no commentary)."
        )
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        try:
            cfg = json.loads(text)
            return cfg, "Gemini suggested visualization."
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                cfg = json.loads(text[start:end+1])
                return cfg, "Gemini suggested visualization (parsed)."
            return None, "Could not parse Gemini response."
    except Exception as e:
        return None, f"Gemini error: {e}"

# ---------------------------------------
# Visualization renderer (seaborn/mpl)
# ---------------------------------------
def render_viz(df, viz_cfg, palette_color=CORAL):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set(style="whitegrid")

    vtype = (viz_cfg.get("type") or "").lower()
    x = viz_cfg.get("x")
    y = viz_cfg.get("y")
    hue = viz_cfg.get("hue")
    agg = (viz_cfg.get("aggregate") or "none").lower()

    plot_df = df.copy()
    try:
        if agg in ["sum", "mean", "count"] and x and y and x in plot_df.columns and (y in plot_df.columns or agg == "count"):
            if agg == "count":
                plot_df = plot_df.groupby(x).size().reset_index(name="count")
                y = "count"
            else:
                plot_df = plot_df.groupby(x)[y].agg(agg).reset_index()

        if vtype == "bar" and x and y:
            if hue:
                sns.barplot(data=plot_df, x=x, y=y, hue=hue)
            else:
                sns.barplot(data=plot_df, x=x, y=y, color=palette_color)
            ax.set_title("Bar chart", fontsize=14)
            plt.xticks(rotation=45)
        elif vtype == "pie" and x:
            counts = plot_df[x].value_counts()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=[palette_color, "#FFDAB9", "#FFC1A6", "#FFA07A", "#B0E0E6", "#8FBC8F"])
            ax.set_title("Pie chart", fontsize=14)
        elif vtype == "scatter" and x and y:
            sns.scatterplot(data=plot_df, x=x, y=y, hue=hue)
            ax.set_title("Scatter plot", fontsize=14)
        elif vtype == "hist" and x:
            sns.histplot(data=plot_df, x=x, hue=hue, color=palette_color, bins=20)
            ax.set_title("Histogram", fontsize=14)
        else:
            ax.text(0.1, 0.5, f"Invalid or insufficient config.\nType: {vtype}\nX: {x}\nY: {y}", fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.1, 0.5, f"Visualization error: {e}", fontsize=12, color="red")
        ax.axis("off")
        return fig

# ---------------------------------------
# Layout
# ---------------------------------------
st.markdown(f"<h2 class='wow-title'>Agent-based Data Visualization</h2>", unsafe_allow_html=True)
st.markdown(f"<div class='wow-sub'>Theme: <span class='wow-accent'>{theme_name}</span></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Inputs")
    gemini_api_key = st.text_input("Gemini API Key (optional)", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")
    use_gemini = st.checkbox("Use Gemini 2.0 Flash for suggestions", value=USE_GEMINI_DEFAULT)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='wow-box'><b>Dataset input</b></div>", unsafe_allow_html=True)
    uploaded_dataset = st.file_uploader("Upload dataset (CSV or JSON)", type=["csv", "json"])
    pasted_dataset = st.text_area("Or paste dataset (CSV or JSON)", height=180, placeholder="Paste CSV or JSON here")

with col2:
    st.markdown("<div class='wow-box'><b>Agents YAML</b></div>", unsafe_allow_html=True)
    uploaded_agents = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
    pasted_agents = st.text_area("Or paste agents.yaml", height=180, placeholder="Paste YAML with agents list")

# Load dataset
df = None
ds_msg = ""
if uploaded_dataset is not None:
    df, ds_msg = detect_format_and_load(uploaded_dataset.read(), filename=uploaded_dataset.name)
elif pasted_dataset.strip():
    df, ds_msg = detect_format_and_load(pasted_dataset)

# Load agents
agents = []
ag_msg = ""
if uploaded_agents is not None:
    agents, ag_msg = parse_agents_yaml(uploaded_agents.read())
elif pasted_agents.strip():
    agents, ag_msg = parse_agents_yaml(pasted_agents)

st.markdown(f"<div class='wow-box'><b>Status</b><br>{ds_msg}<br>{ag_msg}</div>", unsafe_allow_html=True)

agent_names = [a["name"] for a in agents] if agents else []
selected_agent = st.selectbox("Select agent", agent_names) if agent_names else None

# Visualization
if st.button("Visualize", type="primary"):
    if df is None:
        st.error("No dataset loaded.")
    elif not agents:
        st.error("No agents parsed.")
    elif selected_agent is None:
        st.error("Please select an agent.")
    else:
        agent = next((a for a in agents if a["name"] == selected_agent), None)
        if agent is None:
            st.error(f"Selected agent not found: {selected_agent}")
        else:
            viz_cfg = agent.get("visualization") or {}
            gemini_msg = "Gemini not used."
            if use_gemini:
                suggested, gemini_msg = suggest_viz_with_gemini(gemini_api_key, df, agent_context=agent.get("description", ""))
                if suggested:
                    # fill missing keys only
                    for k, v in suggested.items():
                        if viz_cfg.get(k) in [None, "", []]:
                            viz_cfg[k] = v

            # pick a palette color from theme
            palette_color = theme["palette"][0] if theme["palette"] else CORAL
            fig = render_viz(df, viz_cfg, palette_color=palette_color)
            st.pyplot(fig)

            st.markdown("<div class='wow-box'><b>Details</b></div>", unsafe_allow_html=True)
            st.code(json.dumps({
                "agent": agent.get("name"),
                "description": agent.get("description", ""),
                "visualization_config": viz_cfg,
                "gemini_status": gemini_msg
            }, ensure_ascii=False, indent=2), language="json")

# Footer
st.markdown("---")
st.markdown(f"<div class='wow-sub'>Tips</div>", unsafe_allow_html=True)
st.markdown(
    "- Use agents with fields present in your dataset.\n"
    "- If Gemini is enabled, ensure a valid API key is provided.\n"
    "- For categorical distributions, pie/bar works best; for numeric distributions, hist/scatter works best."
)
