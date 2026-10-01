"""
💡 SMART LIGHTING SYSTEM - Intelligent Lighting Demand Prediction & Control
Run with:  python -m streamlit run dashboard.py

Libraries: streamlit, pandas, matplotlib (+ Python standard library only)

Structure
---------
DATA LAYER  -> get_current_conditions(), get_sensor_history()   (replace with ESP32 data)
ML LAYER    -> predict_lighting_demand()                        (replace with trained model)
EXPLAIN     -> explain_prediction()
UI          -> main()
"""

from datetime import date, datetime, time as dtime
from pathlib import Path
import plotly
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import database as db

# ----------------------------------------------------------------------
# One-time theme bootstrap (light/beige theme for sliders, tables, etc.)
# Creates .streamlit/config.toml if missing. Restart Streamlit once after.
# ----------------------------------------------------------------------
def ensure_theme_config():
    cfg = Path(".streamlit") / "config.toml"
    if cfg.exists():
        return
    try:
        cfg.parent.mkdir(exist_ok=True)
        cfg.write_text(
            '[theme]\nbase = "light"\nprimaryColor = "#d99a06"\n'
            'backgroundColor = "#f3ead8"\nsecondaryBackgroundColor = "#fffdf7"\n'
            'textColor = "#2b2b2b"\nfont = "sans serif"\n'
        )
    except OSError:
        pass


ensure_theme_config()
db.init_db()
db.seed_sample_data()  # only fills the DB if it is empty

st.set_page_config(page_title="Smart Lighting System", page_icon="💡", layout="wide",
                   initial_sidebar_state="expanded")

# Palette
BG, CARD, TEXT = "#f3ead8", "#fffdf7", "#2b2b2b"
BORDER, MUTED, AMBER, GREEN = "#c9b79c", "#8b6f4e", "#d99a06", "#22a45d"

st.markdown(
    f"""
<style>
.stApp {{background:{BG}; color:{TEXT};}}
header[data-testid="stHeader"] {{background:transparent;}}
#MainMenu, footer {{visibility:hidden;}}
.block-container {{padding-top:2rem; max-width:1300px;}}
.stMarkdown, .stMarkdown p, [data-testid="stWidgetLabel"] p {{color:{TEXT};}}
[data-testid="stWidgetLabel"] p {{font-weight:700; letter-spacing:.06em;}}
div[data-baseweb="slider"] div[role="slider"] {{background-color:{AMBER} !important;}}
.header {{display:flex; justify-content:space-between; align-items:center; margin-bottom:1.2rem;}}
.title {{font-size:2.1rem; font-weight:800; letter-spacing:.04em; color:{TEXT};}}
.subtitle {{color:{MUTED}; font-size:1.05rem;}}
.online {{background:{CARD}; border:1px solid {BORDER}; border-radius:999px; padding:.5rem 1.1rem;
          font-weight:700; color:{GREEN}; box-shadow:0 2px 8px rgba(120,90,50,.12);}}
.dot {{display:inline-block; width:11px; height:11px; border-radius:50%; background:{GREEN};
       margin-right:.5rem; box-shadow:0 0 8px {GREEN};}}
.section-title {{font-size:1.25rem; font-weight:800; letter-spacing:.05em; margin:1.8rem 0 .8rem 0; color:{TEXT};}}
.card {{background:{CARD}; border:1px solid {BORDER}; border-radius:18px; padding:1.1rem 1.3rem;
        box-shadow:0 4px 14px rgba(120,90,50,.12);}}
.sensor {{text-align:center;}}
.sensor .icon {{font-size:2.2rem;}}
.sensor .label {{color:{MUTED}; font-weight:700; letter-spacing:.12em; font-size:.8rem; margin-top:.2rem;}}
.sensor .value {{font-size:2rem; font-weight:800; color:{TEXT};}}
.row {{display:flex; justify-content:space-between; align-items:center; padding:.55rem 0;
       border-bottom:1px dashed {BORDER}; color:{TEXT};}}
.row:last-child {{border-bottom:none;}}
.chip {{background:#f7e7b8; border:1px solid {AMBER}; border-radius:999px; padding:.1rem .7rem;
        font-weight:800; font-size:.8rem; color:#6b4a00;}}
.card-title {{font-weight:800; letter-spacing:.08em; color:{MUTED}; font-size:.85rem; margin-bottom:.4rem;}}
</style>
""",
    unsafe_allow_html=True,
)

TARGET_LUX = 500
LIGHT_KEYS = ["l1", "l2", "l3", "l4"]
LIGHT_DEFAULTS = {"l1": 75, "l2": 60, "l3": 80, "l4": 70}


# ----------------------------------------------------------------------
# DATA LAYER  (swap for real ESP32 data later)
# ----------------------------------------------------------------------
def get_current_conditions() -> dict:
    """Latest reading stored in the database (the ESP32 will write here later)."""
    row = db.get_latest_reading()
    return {
        "occupancy": int(row["occupancy"]),
        "ambient_lux": int(round(row["ambient_lux"])),
        "temperature": float(row["temperature"]),
        "time": datetime.fromisoformat(row["timestamp"]).strftime("%H:%M"),
    }


def _hour(time_str: str) -> float:
    h, m = time_str.split(":")
    return int(h) + int(m) / 60


def get_sensor_history(limit: int = 16) -> pd.DataFrame:
    """Latest readings from the database, shaped for the charts and table."""
    raw = db.get_history(limit)
    df = pd.DataFrame(
        {
            "Time": pd.to_datetime(raw["timestamp"]).dt.strftime("%H:%M"),
            "Occupancy": raw["occupancy"].astype(int),
            "Ambient Light": raw["ambient_lux"].round().astype(int),
            "Temperature": raw["temperature"].round(1),
        }
    )
    # Use the stored demand if there is one, otherwise predict it
    df["Lighting"] = [
        d if pd.notna(d) else predict_lighting_demand(o, a, t, _hour(tm))
        for d, tm, o, a, t in zip(raw["lighting_demand"], df["Time"], df["Occupancy"],
                                  df["Ambient Light"], df["Temperature"])
    ]
    return df


# ----------------------------------------------------------------------
# ML LAYER  (replace the body with your trained model later)
# ----------------------------------------------------------------------
def _time_factor(hour: float) -> float:
    if hour >= 20 or hour < 6:
        return 1.0
    if hour >= 18:
        return 0.9
    if hour >= 17 or hour < 8:
        return 0.4
    return 0.0


def predict_lighting_demand(occupancy, ambient_lux, temperature, hour) -> float:
    """
    Predict required artificial lighting (0-100 %).

    TEMPORARY simulated logic. Later replace with e.g.:
        model = joblib.load("lighting_model.pkl")
        return float(max(0, min(100, model.predict([[occupancy, ambient_lux, temperature, hour]])[0])))
    """
    if occupancy <= 0:
        return 0.0
    deficit = max(0.0, TARGET_LUX - ambient_lux) / TARGET_LUX
    occ = min(occupancy / 4, 1.0)
    score = 0.6 * deficit + 0.2 * occ + 0.2 * _time_factor(hour)
    return float(max(0, min(100, round(score * 100))))


# ----------------------------------------------------------------------
# EXPLANATION
# ----------------------------------------------------------------------
def explain_prediction(c: dict, demand: float):
    occ, lux, temp, hr = c["occupancy"], c["ambient_lux"], c["temperature"], _hour(c["time"])
    occ_lvl = "NONE" if occ == 0 else "LOW" if occ == 1 else "MEDIUM" if occ == 2 else "HIGH"
    lux_lvl = "LOW" if lux < 200 else "MEDIUM" if lux < 450 else "HIGH"
    time_lvl = "NIGHT" if (hr >= 18 or hr < 6) else "MORNING" if hr < 12 else "AFTERNOON"
    temp_lvl = "COOL" if temp < 22 else "NORMAL" if temp <= 29 else "WARM"
    factors = [("👥", "Occupancy", occ_lvl), ("☀️", "Ambient Light", lux_lvl),
               ("🕐", "Time", time_lvl), ("🌡️", "Temperature", temp_lvl)]
    if demand >= 70:
        msg = "Higher artificial lighting is required"
    elif demand >= 40:
        msg = "Moderate artificial lighting is required"
    elif demand > 0:
        msg = "Only light supplementary lighting is needed"
    else:
        msg = "Lights can stay off (no occupancy or enough daylight)"
    return factors, msg


# ----------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------
def section_title(text):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def sensor_card(icon, label, value):
    return (f"<div class='card sensor'><div class='icon'>{icon}</div>"
            f"<div class='label'>{label}</div><div class='value'>{value}</div></div>")


def level_name(b):
    return "OFF" if b <= 5 else "ALMOST OFF" if b <= 20 else "LOW" if b <= 50 else "MEDIUM" if b <= 80 else "HIGH"


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def light_html(label, b, left, top):
    a = b / 100
    halo = (f"<div style='position:absolute;left:50%;top:45%;width:240px;height:240px;"
            f"transform:translate(-50%,-50%);pointer-events:none;"
            f"background:radial-gradient(circle,rgba(255,193,7,{0.7 * a:.2f}) 0%,rgba(255,193,7,0) 70%);'></div>")
    bulb = (f"<div style='position:relative;font-size:2.6rem;opacity:{0.3 + 0.7 * a:.2f};"
            f"filter:drop-shadow(0 0 {4 + 20 * a:.0f}px rgba(255,170,0,{a:.2f}));'>💡</div>")
    txt = (f"<div style='position:relative;font-weight:800;color:{TEXT};'>{label}</div>"
           f"<div style='position:relative;font-size:.75rem;color:{MUTED};font-weight:700;'>{b}% · {level_name(b)}</div>")
    return (f"<div style='position:absolute;left:{left}%;top:{top}%;transform:translate(-50%,-50%);"
            f"text-align:center;width:130px;'>{halo}{bulb}{txt}</div>")


def room_html(levels, occupancy, ambient):
    avg = sum(levels) / len(levels)
    r, g, bl = _lerp((214, 203, 182), (255, 248, 228), avg / 100)
    sky = min(ambient / 600, 1.0)
    window = (f"<div style='position:absolute;top:0;left:12%;right:12%;height:44px;text-align:center;"
              f"line-height:44px;font-weight:800;letter-spacing:.2em;color:{TEXT};"
              f"border:3px solid {MUTED};border-top:none;border-radius:0 0 12px 12px;"
              f"background:linear-gradient(180deg,rgba(135,185,220,{0.25 + 0.55 * sky:.2f}),rgba(255,250,235,.6));'>"
              f"🪟 WINDOW · {ambient} lux</div>")
    door = (f"<div style='position:absolute;bottom:0;left:50%;transform:translateX(-50%);width:150px;height:38px;"
            f"text-align:center;line-height:38px;font-weight:800;letter-spacing:.2em;color:#fff;"
            f"background:{MUTED};border-radius:12px 12px 0 0;'>🚪 DOOR</div>")
    people = "👤" * min(occupancy, 5) if occupancy > 0 else "∅"
    who = f"{occupancy} OCCUPANT{'S' if occupancy != 1 else ''}" if occupancy > 0 else "EMPTY ROOM"
    occupant = (f"<div style='position:absolute;left:50%;top:47%;transform:translate(-50%,-50%);text-align:center;'>"
                f"<div style='font-size:2.4rem;'>{people}</div>"
                f"<div style='font-weight:800;letter-spacing:.12em;font-size:.8rem;color:{TEXT};'>{who}</div></div>")
    lights = (light_html("L1", levels[0], 25, 26) + light_html("L2", levels[1], 75, 26)
              + light_html("L3", levels[2], 25, 68) + light_html("L4", levels[3], 75, 68))
    return (f"<div style='position:relative;height:500px;border-radius:22px;border:4px solid {MUTED};"
            f"background:rgb({r},{g},{bl});overflow:hidden;box-shadow:0 6px 18px rgba(120,90,50,.2);'>"
            f"{window}{lights}{occupant}{door}</div>")


def gauge_html(pct):
    deg = pct * 3.6
    return (f"<div style='width:210px;height:210px;border-radius:50%;margin:.4rem auto;"
            f"background:conic-gradient({AMBER} 0deg {deg}deg,#eadfca {deg}deg 360deg);"
            f"display:flex;align-items:center;justify-content:center;'>"
            f"<div style='width:164px;height:164px;border-radius:50%;background:{CARD};"
            f"display:flex;flex-direction:column;align-items:center;justify-content:center;'>"
            f"<div style='font-size:3.1rem;font-weight:800;color:{TEXT};line-height:1;'>{pct:.0f}%</div>"
            f"<div style='font-size:.75rem;color:{MUTED};font-weight:700;margin-top:.3rem;'>"
            f"Recommended<br>Brightness</div></div></div>")


def make_chart(df, col, title, ylabel, color, ylim=None, fill=False, now="19:00"):
    fig = plt.figure(figsize=(6.4, 3.6), dpi=110)      # same size for every chart
    fig.patch.set_facecolor(CARD)
    ax = fig.add_axes([0.12, 0.26, 0.85, 0.56])        # fixed plot area -> all charts identical
    ax.set_facecolor(CARD)
    x = list(range(len(df)))
    ax.plot(x, df[col], color=color, lw=2.5, marker="o", ms=6, mfc=CARD, mew=2)
    if fill:
        ax.fill_between(x, df[col], color=color, alpha=0.18)
    times = list(df["Time"])
    if now in times:
        i = len(times) - 1 - times[::-1].index(now)
        ax.axvline(i, color=MUTED, ls="--", lw=1)
        ax.text(i, 1.03, "now", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(times, fontsize=8, color=TEXT, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=8, colors=TEXT)
    ax.set_xlabel("Time", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.margins(y=0.15)
    if ylim:
        ax.set_ylim(*ylim)
    fig.text(0.03, 0.96, title, fontsize=12, fontweight="bold", color=TEXT, va="top")
    ax.grid(axis="y", color="#eadfca", lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BORDER)
    return fig


def apply_prediction(value):
    for k in LIGHT_KEYS:
        st.session_state[k] = int(value)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
@st.cache_data
def load_raw_stream():
    """Read the RAW sensor log and clean it (drop dropouts, spikes and -127 faults)."""
    db.ensure_raw_csv()
    raw = pd.read_csv(db.RAW_CSV)
    for c in ("occupancy", "ambient_lux", "temperature"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    ok = (
        raw.notna().all(axis=1)
        & raw["occupancy"].between(0, 50)
        & raw["ambient_lux"].between(0, 2000)
        & raw["temperature"].between(-10, 60)
    )
    return raw[ok].to_dict("records")


def advance_stream():
    """Insert the next cleaned raw reading into the database (like an ESP32 would)."""
    rows = load_raw_stream()
    i = st.session_state.live_idx
    if i >= len(rows):
        st.session_state.live = False
        return
    r = rows[i]
    ts = f"{date.today().isoformat()}T{r['time']}:00"
    db.insert_reading(r["occupancy"], r["ambient_lux"], r["temperature"], timestamp=ts)
    st.session_state.live_idx = i + 1
    if st.session_state.live_idx >= len(rows):
        st.session_state.live = False


def start_live():
    rows = load_raw_stream()
    idx = st.session_state.live_idx
    if idx == 0 or idx >= len(rows):          # fresh start (resume if paused mid-way)
        db.clear_readings()
        st.session_state.live_idx = 0
        advance_stream()                      # first reading right away
    st.session_state.live = True


def pause_live():
    st.session_state.live = False


def stop_live():
    st.session_state.live = False
    st.session_state.live_idx = 0
    db.reset_db()


def control_panel():
    """Sidebar: switch datasets or add readings without using the terminal."""
    with st.sidebar:
        for k, v in (("live", False), ("live_idx", 0), ("tick", 0), ("interval", 2)):
            st.session_state.setdefault(k, v)

        st.markdown("### 📡 Live Monitoring")
        st.caption("Streams a raw sensor log reading by reading, like the ESP32 will.")
        rows = load_raw_stream()
        speeds = {"Slow (3 s)": 3, "Normal (2 s)": 2, "Fast (1 s)": 1}
        st.session_state.interval = speeds[st.selectbox("Speed per reading", list(speeds), index=1)]
        b1, b2 = st.columns(2)
        if st.session_state.live:
            b1.button("⏸ Pause", on_click=pause_live)
        else:
            b1.button("▶ Start", on_click=start_live)
        b2.button("⏹ Stop", on_click=stop_live)
        idx = st.session_state.live_idx
        st.progress(min(idx / len(rows), 1.0))
        state = "🔴 Streaming" if st.session_state.live else ("✅ Finished" if idx >= len(rows) else "⏸ Idle / paused")
        st.caption(f"{state} · reading {idx}/{len(rows)}")

        @st.fragment(run_every=1 if st.session_state.live else None)
        def _ticker():
            if not st.session_state.live:
                return
            st.session_state.tick += 1
            if st.session_state.tick % st.session_state.interval == 0:
                advance_stream()
                st.rerun()          # refresh the whole dashboard with the new reading

        _ticker()
        st.markdown("---")
        st.markdown("### 🎛️ Data Control")
        st.caption("Simulates what the ESP32 will send later.")
        name = st.selectbox("Dataset", list(db.DATASETS.keys()))
        if st.button("📂 Load dataset"):
            st.session_state.live = False
            st.session_state.live_idx = 0
            db.load_dataset(name)

        st.markdown("---")
        st.markdown("**Add a reading**")
        occ = st.number_input("Occupancy (people)", 0, 50, 3)
        lux = st.number_input("Ambient light (lux)", 0, 2000, 150, step=10)
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, 27.0, step=0.5)
        t = st.time_input("Time of reading", value=dtime(19, 0))
        if st.button("➕ Add reading"):
            st.session_state.live = False
            ts = datetime.combine(date.today(), t).isoformat(timespec="seconds")
            db.insert_reading(occ, lux, temp, timestamp=ts)

        st.markdown("---")
        if st.button("♻️ Reset to demo data"):
            st.session_state.live = False
            st.session_state.live_idx = 0
            db.reset_db()


def main():
    control_panel()
    for k in LIGHT_KEYS:
        st.session_state.setdefault(k, LIGHT_DEFAULTS[k])
    st.session_state.setdefault("auto_mode", True)

    cur = get_current_conditions()
    hist = get_sensor_history()
    demand = predict_lighting_demand(cur["occupancy"], cur["ambient_lux"], cur["temperature"], _hour(cur["time"]))
    recommended = int(round(demand))

    # 1. HEADER
    st.markdown(
        "<div class='header'><div><div class='title'>💡 SMART LIGHTING SYSTEM</div>"
        "<div class='subtitle'>Intelligent Lighting Demand Prediction &amp; Control</div></div>"
        "<div class='online'><span class='dot'></span>SYSTEM ONLINE" + (" · 🔴 LIVE" if st.session_state.get("live") else "") + "</div></div>",
        unsafe_allow_html=True,
    )

    # 2. LIVE SENSOR CARDS
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(sensor_card("👥", "OCCUPANCY", f"{cur['occupancy']} People"), unsafe_allow_html=True)
    c2.markdown(sensor_card("☀️", "AMBIENT LIGHT", f"{cur['ambient_lux']} Lux"), unsafe_allow_html=True)
    c3.markdown(sensor_card("🌡️", "TEMPERATURE", f"{cur['temperature']:.0f} °C"), unsafe_allow_html=True)
    c4.markdown(sensor_card("🕐", "CURRENT TIME", cur["time"]), unsafe_allow_html=True)

    # 3 + 4. ROOM + CONTROLS
    section_title("🏠 ROOM LIGHTING CONTROL")
    room_col, ctrl_col = st.columns([2.2, 1])

    with ctrl_col:
        st.markdown("<div class='card-title'>LIGHT CONTROLS</div>", unsafe_allow_html=True)
        auto = st.toggle("🤖 Automatic control (follow AI prediction)", key="auto_mode")
        if auto:                      # lights follow the prediction on every new reading
            for k in LIGHT_KEYS:
                st.session_state[k] = recommended
        for i, k in enumerate(LIGHT_KEYS, start=1):
            st.slider(f"LIGHT {i}", 0, 100, key=k, format="%d%%", disabled=auto)
        st.button("Apply Predicted Brightness", on_click=apply_prediction,
                  args=(recommended,), disabled=auto)
        levels = [st.session_state[k] for k in LIGHT_KEYS]
        avg = round(sum(levels) / 4)
        on = avg > 0
        st.markdown(
            f"<div class='card' style='margin-top:.8rem;'>"
            f"<div class='row'><span class='card-title' style='margin:0'>LED STATUS</span>"
            f"<b style='color:{GREEN if on else MUTED};'>● {'ON' if on else 'OFF'}</b></div>"
            f"<div class='row'><span class='card-title' style='margin:0'>AVERAGE ROOM BRIGHTNESS</span>"
            f"<b style='font-size:1.6rem;'>{avg}%</b></div></div>",
            unsafe_allow_html=True,
        )

    with room_col:
        st.markdown(room_html(levels, cur["occupancy"], cur["ambient_lux"]), unsafe_allow_html=True)

    # 5 + 6. PREDICTION + WHY
    section_title("🔮 LIGHTING DEMAND PREDICTION")
    p1, p2, p3 = st.columns([1, 1, 1.1])
    with p1:
        st.markdown(f"<div class='card'>{gauge_html(demand)}</div>", unsafe_allow_html=True)
    with p2:
        st.markdown(
            f"<div class='card'><div class='card-title'>PREDICTION SUMMARY</div>"
            f"<div class='row'><span>Predicted Demand</span><b>{demand:.0f}%</b></div>"
            f"<div class='row'><span>Recommended Brightness</span><b>{recommended}%</b></div>"
            f"<div class='row'><span>LED Status</span>"
            f"<b style='color:{GREEN if recommended > 0 else MUTED};'>{'ON' if recommended > 0 else 'OFF'}</b></div>"
            f"<div class='row'><span>Control Mode</span><b>{'AUTOMATIC' if auto else 'MANUAL'}</b></div></div>",
            unsafe_allow_html=True,
        )
    with p3:
        factors, msg = explain_prediction(cur, demand)
        rows = "".join(
            f"<div class='row'><span>{ic} {nm}</span><span class='chip'>{lv}</span></div>" for ic, nm, lv in factors
        )
        st.markdown(
            f"<div class='card'><div class='card-title'>WHY THIS PREDICTION?</div>{rows}"
            f"<div style='margin-top:.7rem;font-weight:800;color:#6b4a00;'>→ {msg}</div></div>",
            unsafe_allow_html=True,
        )

    # 7. SENSOR ANALYTICS
    section_title("📊 SENSOR ANALYTICS")
    g1, g2 = st.columns(2)
    with g1:
        fig = make_chart(hist, "Occupancy", "Occupancy vs Time", "Number of occupants", "#7a5c3a", now=cur["time"])
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        fig = make_chart(hist, "Ambient Light", "Ambient Light vs Time", "Ambient light (lux)", "#e0a800", now=cur["time"])
        st.pyplot(fig)
        plt.close(fig)
    g3, g4 = st.columns(2)
    with g3:
        fig = make_chart(hist, "Temperature", "Temperature vs Time", "Temperature (°C)", "#c0583a", now=cur["time"])
        st.pyplot(fig)
        plt.close(fig)
    with g4:
        fig = make_chart(hist, "Lighting", "Lighting Demand vs Time", "Lighting demand (%)", AMBER,
                         ylim=(0, 100), fill=True, now=cur["time"])
        st.pyplot(fig)
        plt.close(fig)

    # 8. SYSTEM STATUS
    section_title("⚙️ SYSTEM STATUS")
    items = [("ESP32", "Connected"), ("Sensors", "Active"), ("Prediction Model", "Ready"), ("LED Controller", "Active")]
    for col, (name, state) in zip(st.columns(4), items):
        col.markdown(
            f"<div class='card sensor'><div class='label'>{name.upper()}</div>"
            f"<div style='font-weight:800;color:{GREEN};margin-top:.3rem;'>● {state}</div></div>",
            unsafe_allow_html=True,
        )

    # 9. DATA TABLE
    st.write("")
    with st.expander("📋 VIEW SENSOR DATA"):
        st.dataframe(hist, hide_index=True)

    st.caption("Prototype · simulated sensor data and prediction · ready for ESP32 + ML integration")


main()
