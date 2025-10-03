# streamlit_sports_demo_v_final.py
"""
Refactored Sports Data Analyst Streamlit app.

Features preserved:
- Simulate or upload per-second tracking + events
- Configure match start, seed, players, minutes/half, event density, pitch zones
- Player roles + role filter + robust player selector with session-state
- KPIs (distance, speed, sprints, accelerations), zone distances, heatmaps
- Inferred pass network, simple xG for shots
- Optional animated movement (downsampled)
- Overlay trajectories with faint trails and interactive legend
- Export overlay + heatmap PNGs into ZIP + one-page PDF generator
- Rename player (propagates to tracking & events in-session) and recompute KPIs
"""
from typing import List, Tuple, Optional, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import zipfile
from datetime import datetime, timedelta, date, time
import json

st.set_page_config(page_title="Sports Data Analyst", layout="wide")


# ---------------------------
# Simulation helpers
# ---------------------------
def simulate_match_data(players: Optional[List[str]] = None,
                        minutes_per_half: int = 15,
                        seed: int = 42,
                        start_datetime: Optional[datetime] = None,
                        pass_range: Tuple[int, int] = (5, 15),
                        shot_range: Tuple[int, int] = (0, 5),
                        pass_success_prob: float = 0.85,
                        shot_success_prob: float = 0.4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate per-second tracking for each player and random events (pass/shot)."""
    if players is None or len(players) == 0:
        players = [f"Player_{i+1:02d}" for i in range(11)]
    np.random.seed(int(seed) if seed is not None else 42)

    total_seconds = int(minutes_per_half * 60)
    all_tracking = []
    all_events = []

    if start_datetime is None:
        start_datetime = datetime.now().replace(microsecond=0)

    for half in [1, 2]:
        start_time = start_datetime + timedelta(minutes=(half - 1) * minutes_per_half)
        timestamps = pd.date_range(start=start_time, periods=total_seconds, freq="1S")

        for p in players:
            position_factor = np.random.uniform(0.4, 1.0)
            x = np.cumsum(np.random.normal(loc=0.0, scale=0.5 * position_factor, size=total_seconds)) + 52.5 + np.random.uniform(-6, 6)
            y = np.cumsum(np.random.normal(loc=0.0, scale=0.4 * position_factor, size=total_seconds)) + 34 + np.random.uniform(-4, 4)
            x = np.clip(x, 0, 105)
            y = np.clip(y, 0, 68)

            df = pd.DataFrame({
                "timestamp": timestamps.astype(str),
                "player": p,
                "x": np.round(x, 3),
                "y": np.round(y, 3),
                "half": half
            })
            all_tracking.append(df)

            n_passes = int(np.random.randint(pass_range[0], pass_range[1] + 1)) if pass_range[1] >= pass_range[0] else 0
            n_shots = int(np.random.randint(shot_range[0], shot_range[1] + 1)) if shot_range[1] >= shot_range[0] else 0

            if len(timestamps) > 0 and n_passes > 0:
                pass_times = np.random.choice(timestamps, n_passes, replace=False)
                for t in pass_times:
                    all_events.append({
                        "timestamp": str(t),
                        "player": p,
                        "event": "pass",
                        "success": int(np.random.choice([0, 1], p=[1 - pass_success_prob, pass_success_prob])),
                        "x": round(np.random.uniform(0, 105), 3),
                        "y": round(np.random.uniform(0, 68), 3),
                        "half": half
                    })
            if len(timestamps) > 0 and n_shots > 0:
                shot_times = np.random.choice(timestamps, n_shots, replace=False)
                for t in shot_times:
                    all_events.append({
                        "timestamp": str(t),
                        "player": p,
                        "event": "shot",
                        "success": int(np.random.choice([0, 1], p=[1 - shot_success_prob, shot_success_prob])),
                        "x": round(np.random.uniform(0, 105), 3),
                        "y": round(np.random.uniform(0, 68), 3),
                        "half": half
                    })

    tracking_df = pd.concat(all_tracking).reset_index(drop=True) if all_tracking else pd.DataFrame()
    events_df = pd.DataFrame(all_events).reset_index(drop=True) if all_events else pd.DataFrame()
    return tracking_df, events_df


# ---------------------------
# Processing / KPIs
# ---------------------------
def process_tracking(tracking: pd.DataFrame) -> pd.DataFrame:
    """Compute step distances, speeds, accelerations and cumulative distance per player."""
    if tracking is None or tracking.empty:
        return pd.DataFrame()
    t = tracking.copy()
    t['timestamp'] = pd.to_datetime(t['timestamp'])
    t = t.sort_values(['player', 'timestamp']).reset_index(drop=True)
    t[['x_prev', 'y_prev', 't_prev']] = t.groupby('player')[['x', 'y', 'timestamp']].shift(1)
    t['dt'] = (t['timestamp'] - t['t_prev']).dt.total_seconds().fillna(0)
    t['dx'] = (t['x'] - t['x_prev']).fillna(0)
    t['dy'] = (t['y'] - t['y_prev']).fillna(0)
    t['step_distance_m'] = np.sqrt(t['dx'] ** 2 + t['dy'] ** 2)
    t['speed_m_s'] = t['step_distance_m'] / t['dt'].replace(0, 1)
    t['speed_m_s'] = t['speed_m_s'].replace([np.inf, -np.inf], 0).fillna(0)
    t['speed_prev'] = t.groupby('player')['speed_m_s'].shift(1).fillna(0)
    t['accel_m_s2'] = (t['speed_m_s'] - t['speed_prev']) / t['dt'].replace(0, 1)
    t['accel_m_s2'] = t['accel_m_s2'].replace([np.inf, -np.inf], 0).fillna(0)
    t['cumulative_distance_m'] = t.groupby('player')['step_distance_m'].cumsum()
    return t


def compute_kpis(tracking_processed: pd.DataFrame,
                 events_df: Optional[pd.DataFrame] = None,
                 sprint_threshold: float = 7.0,
                 accel_threshold: float = 2.0) -> pd.DataFrame:
    """Aggregate KPIs per player."""
    if tracking_processed is None or tracking_processed.empty:
        return pd.DataFrame()
    kpi_list = []
    for p, g in tracking_processed.groupby('player'):
        total_distance = g['step_distance_m'].sum()
        max_speed = g['speed_m_s'].max() if not g['speed_m_s'].empty else 0.0
        avg_speed = g['speed_m_s'].mean() if not g['speed_m_s'].empty else 0.0
        hi_distance = g.loc[g['speed_m_s'] >= sprint_threshold, 'step_distance_m'].sum()
        hi_seconds = int((g['speed_m_s'] >= sprint_threshold).sum())
        hi_pct = round((hi_seconds / len(g) * 100) if len(g) > 0 else 0, 2)
        is_sprint = (g['speed_m_s'] >= sprint_threshold).astype(int).values
        sprint_count = int(((np.diff(np.pad(is_sprint, (1, 0), 'constant')) == 1)).sum())
        accel_count = int((g['accel_m_s2'] >= accel_threshold).sum())
        max_accel = float(g['accel_m_s2'].max()) if not g['accel_m_s2'].empty else 0.0
        avg_accel = float(g['accel_m_s2'].mean()) if not g['accel_m_s2'].empty else 0.0
        kpi = {
            "player": p,
            "total_distance_m": round(total_distance, 1),
            "max_speed_m_s": round(max_speed, 2),
            "avg_speed_m_s": round(avg_speed, 2),
            "hi_distance_m": round(hi_distance, 1),
            "hi_seconds": hi_seconds,
            "hi_pct": hi_pct,
            "sprint_count_est": sprint_count,
            "accel_count_gt2m_s2": accel_count,
            "max_accel_m_s2": round(max_accel, 2),
            "avg_accel_m_s2": round(avg_accel, 2),
            "minutes": int(len(g) / 60)
        }
        kpi_list.append(kpi)
    kpis = pd.DataFrame(kpi_list)
    if events_df is not None and not events_df.empty:
        ev_agg = events_df.groupby('player').agg(
            passes=('event', lambda x: (x == 'pass').sum()),
            shots=('event', lambda x: (x == 'shot').sum())
        ).reset_index()
        kpis = kpis.merge(ev_agg, on='player', how='left')
        kpis['passes'] = kpis['passes'].fillna(0).astype(int)
        kpis['shots'] = kpis['shots'].fillna(0).astype(int)
    else:
        kpis['passes'] = 0
        kpis['shots'] = 0
    return kpis


def compute_zone_distances(tracking_processed: pd.DataFrame, nx: int = 3, ny: int = 3,
                           pitch_length: int = 105, pitch_width: int = 68) -> pd.DataFrame:
    """Compute distance per tactical zone for each player."""
    if tracking_processed is None or tracking_processed.empty:
        return pd.DataFrame()
    x_bins = np.linspace(0, pitch_length, nx + 1)
    y_bins = np.linspace(0, pitch_width, ny + 1)
    zone_rows = []
    for p, g in tracking_processed.groupby('player'):
        g = g.copy()
        g['x_pos'] = g['x_prev'].fillna(g['x'])
        g['y_pos'] = g['y_prev'].fillna(g['y'])
        x_zone = np.digitize(g['x_pos'], x_bins) - 1
        y_zone = np.digitize(g['y_pos'], y_bins) - 1
        x_zone = np.clip(x_zone, 0, nx - 1)
        y_zone = np.clip(y_zone, 0, ny - 1)
        g['zone'] = x_zone + y_zone * nx
        zone_dist = g.groupby('zone')['step_distance_m'].sum().reindex(range(nx * ny), fill_value=0)
        row = {"player": p}
        for zid, dist in zone_dist.items():
            row[f"zone_{zid}_dist_m"] = round(dist, 1)
        zone_rows.append(row)
    return pd.DataFrame(zone_rows)


# ---------------------------
# Pass network, xG, animation
# ---------------------------
def infer_pass_network(events_df: pd.DataFrame, tracking_df: pd.DataFrame, max_radius: float = 10.0) -> pd.DataFrame:
    """Infer pass edges by proximity of players at pass timestamps (simple heuristic)."""
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=['from', 'to', 'count'])
    events_df = events_df.copy()
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    tracking_df = tracking_df.copy()
    tracking_df['timestamp'] = pd.to_datetime(tracking_df['timestamp'])

    pass_events = events_df[events_df['event'] == 'pass']
    edges = {}
    for _, ev in pass_events.iterrows():
        t = ev['timestamp']
        passer = ev['player']
        window = tracking_df[tracking_df['timestamp'] == t]
        if window.empty:
            t_floor = t.floor('S')
            window = tracking_df[tracking_df['timestamp'] == t_floor]
            if window.empty:
                continue
        passer_pos = window[window['player'] == passer]
        if passer_pos.empty:
            continue
        px, py = float(passer_pos.iloc[0]['x']), float(passer_pos.iloc[0]['y'])
        others = window[window['player'] != passer].copy()
        if others.empty:
            continue
        others['dist'] = np.sqrt((others['x'] - px) ** 2 + (others['y'] - py) ** 2)
        nearest = others.loc[others['dist'].idxmin()]
        if nearest['dist'] <= max_radius:
            key = (passer, nearest['player'])
            edges[key] = edges.get(key, 0) + 1
    rows = [{'from': k[0], 'to': k[1], 'count': v} for k, v in edges.items()]
    return pd.DataFrame(rows)


def compute_xg_for_shots(events_df: pd.DataFrame, goal_x: int = 105, goal_y_center: float = 34.0) -> pd.DataFrame:
    """Toy xG model using distance and lateral displacement."""
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    ev = events_df.copy()
    ev = ev[ev['event'] == 'shot'].copy()
    if ev.empty:
        return pd.DataFrame()
    ev['dist_to_goal'] = np.sqrt((ev['x'] - goal_x) ** 2 + (ev['y'] - goal_y_center) ** 2)
    ev['angle'] = np.abs(ev['y'] - goal_y_center)
    a, b, c = 3.0, -0.12, -0.04
    ev['xG'] = 1 / (1 + np.exp(-(a + b * ev['dist_to_goal'] + c * ev['angle'])))
    ev['xG'] = ev['xG'].clip(0, 1)
    return ev


def create_animation_fig(tracking_df: pd.DataFrame, players_to_show: Optional[List[str]] = None, max_frames: int = 300) -> Optional[go.Figure]:
    """Create a Plotly animation figure showing players as moving markers."""
    if tracking_df is None or tracking_df.empty:
        return None
    t = tracking_df.copy()
    t['timestamp'] = pd.to_datetime(t['timestamp'])
    if players_to_show is not None:
        t = t[t['player'].isin(players_to_show)]
    t = t.sort_values('timestamp')
    unique_times = t['timestamp'].unique()
    if len(unique_times) == 0:
        return None
    if len(unique_times) > max_frames:
        idx = np.linspace(0, len(unique_times) - 1, max_frames).astype(int)
        times_sampled = unique_times[idx]
    else:
        times_sampled = unique_times
    frames = []
    for tm in times_sampled:
        df_tm = t[t['timestamp'] == tm]
        frame = go.Frame(data=[go.Scatter(x=df_tm['x'],
                                          y=df_tm['y'],
                                          mode='markers+text',
                                          text=df_tm['player'],
                                          textposition='top center',
                                          marker=dict(size=8))], name=str(tm))
        frames.append(frame)
    df0 = t[t['timestamp'] == times_sampled[0]] if len(times_sampled) > 0 else t.iloc[:0]
    fig = go.Figure(frames=frames)
    fig.add_trace(go.Scatter(x=df0['x'], y=df0['y'], mode='markers+text', text=df0['player'], textposition='top center', marker=dict(size=8)))
    fig.update_layout(title='Player movement (animated)', xaxis=dict(range=[0, 105]), yaxis=dict(range=[0, 68]),
                      updatemenus=[dict(type='buttons', showactive=False, y=1.05, x=1.15, xanchor='right', yanchor='top',
                                        buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]),
                                                 dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0}, 'mode': 'immediate', 'transition': {'duration': 0}}])])])
    return fig


# ---------------------------
# PDF / ZIP helpers
# ---------------------------
def create_one_page_pdf_bytes(kpis_df: pd.DataFrame, heatmap_png_bytes: bytes, zone_png_bytes: bytes) -> bytes:
    """Render a simple one-page PDF with a small KPI list and up to two images."""
    dpi = 150
    a4_w_in, a4_h_in = 8.27, 11.69
    a4_w_px = int(a4_w_in * dpi)
    a4_h_px = int(a4_h_in * dpi)
    canvas = Image.new("RGB", (a4_w_px, a4_h_px), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()
    draw.text((40, 30), "Kurzbericht: Sports Data Analyst (final)", fill="black", font=font_title)
    y_text = 80
    if kpis_df is None:
        kpis_df = pd.DataFrame()
    for _, row in kpis_df.iterrows():
        line = f"{row['player']}: Dist {row.get('total_distance_m', 0)} m | MaxSpd {row.get('max_speed_m_s', 0)} m/s | Sprints {row.get('sprint_count_est', 0)} | HI% {row.get('hi_pct', 0)}%"
        draw.text((40, y_text), line, fill="black", font=font_small)
        y_text += 18
        if y_text > a4_h_px - 200:
            break
    try:
        heat = Image.open(BytesIO(heatmap_png_bytes))
        heat.thumbnail((420, 300))
        canvas.paste(heat, (420, 110))
    except Exception:
        pass
    try:
        zone = Image.open(BytesIO(zone_png_bytes))
        zone.thumbnail((420, 220))
        canvas.paste(zone, (420, 420))
    except Exception:
        pass
    buf = BytesIO()
    canvas.save(buf, format="PDF", resolution=dpi)
    buf.seek(0)
    return buf.getvalue()


def make_zip_bytes(files_dict: Dict[str, bytes]) -> bytes:
    """Create a ZIP bytes object from a dict of filename->bytes."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files_dict.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------
# Utilities
# ---------------------------
def normalize_player_name(name: str, titlecase: bool = True) -> str:
    """Normalize spacing and optionally title-case an input name."""
    if not isinstance(name, str):
        return name
    n = name.strip()
    return n.title() if titlecase else n


# ---------------------------
# Session-state init for consistent keys
# ---------------------------
def ensure_session_state_defaults():
    st.session_state.setdefault('players_master', [])
    st.session_state.setdefault('last_tracking', pd.DataFrame())
    st.session_state.setdefault('last_events', pd.DataFrame())
    st.session_state.setdefault('kpis_full', pd.DataFrame())
    st.session_state.setdefault('shots_xg', pd.DataFrame())
    st.session_state.setdefault('player_sel', None)


ensure_session_state_defaults()


# ---------------------------
# Sidebar UI & inputs
# ---------------------------
st.title("Shine Jose Sports Data Analyst")
st.markdown("Simulate or upload match data. Improvements: player sync, rename, combined movement export.")
st.markdown("Let's do it...")

with st.sidebar:
    st.header("Simulation Controls")
    minutes_per_half = st.number_input("Minutes per Half", min_value=1, max_value=90, value=15)
    seed = st.number_input("Random Seed (int)", value=42, step=1)

    st.markdown("**Match start (Europe/Berlin timezone)**")
    start_date = st.date_input("Start date", value=date.today())
    start_time = st.time_input("Start time", value=datetime.now().time().replace(microsecond=0))
    start_datetime = datetime.combine(start_date, start_time)

    st.markdown("**Players (one per line). Optionally append ,GK for goalkeeper**")
    default_players = "\n".join([f"Player_{i+1:02d}" + (",GK" if i == 0 else "") for i in range(11)])
    players_text = st.text_area("Player names (one per line)", value=default_players, height=180)

    normalize_names = st.checkbox("Normalize player names (Title Case)", value=True)

    st.markdown("**Event density per player per half**")
    pass_min = st.slider("Min passes per player per half", 0, 200, 10)
    pass_max = st.slider("Max passes per player per half", 0, 400, 25)
    if pass_max < pass_min:
        pass_max = pass_min
    shot_min = st.slider("Min shots per player per half", 0, 50, 0)
    shot_max = st.slider("Max shots per player per half", 0, 100, 3)
    if shot_max < shot_min:
        shot_max = shot_min

    st.markdown("**Event success probabilities**")
    pass_success_prob = st.slider("Pass success probability", 0.0, 1.0, 0.85)
    shot_success_prob = st.slider("Shot success probability", 0.0, 1.0, 0.4)

    st.markdown('---')
    st.header("Pitch / Zones / Visualization")
    nx = st.number_input("Zones across (nx)", value=3, min_value=1, max_value=6)
    ny = st.number_input("Zones down (ny)", value=3, min_value=1, max_value=6)
    enable_animation = st.checkbox("Enable animation (may be slow on long matches)", value=False)
    max_anim_frames = st.number_input("Max animation frames", min_value=50, max_value=1000, value=300)

    st.markdown('---')
    st.header("Upload External Data (optional)")
    uploaded_tracking = st.file_uploader("Upload tracking CSV (timestamp, player, x, y, half)", type=["csv"])
    uploaded_events = st.file_uploader("Upload events CSV (timestamp, player, event, success, x, y, half)", type=["csv"])

    st.markdown('---')
    st.markdown("**Player management**")
    if st.button('Sync players (force sidebar list into app)'):
        raw_lines = [p.strip() for p in players_text.splitlines() if p.strip()]
        parsed = []
        for line in raw_lines:
            if ',' in line:
                name, _ = [x.strip() for x in line.split(',', 1)]
                parsed.append(normalize_player_name(name, titlecase=normalize_names))
            else:
                parsed.append(normalize_player_name(line, titlecase=normalize_names))
        st.session_state['players_master'] = parsed
        st.success('Players synced into session. Now press Generate / Run Analysis')

    st.markdown("**Current players in app session**")
    st.write(st.session_state.get('players_master', []))

    generate_clicked = st.button('Generate / Run Analysis')


# ---------------------------
# Core recompute functions (store results in session state)
# ---------------------------
def recompute_kpis_from_session():
    """Recompute KPIs and supporting tables using tracking/events stored in session_state."""
    tracking = st.session_state.get('last_tracking', pd.DataFrame()).copy()
    events = st.session_state.get('last_events', pd.DataFrame()).copy()

    if not tracking.empty:
        tracking['player'] = tracking['player'].astype(str).str.strip()
        if normalize_names:
            tracking['player'] = tracking['player'].apply(lambda x: normalize_player_name(x, titlecase=True))
    if not events.empty:
        events['player'] = events['player'].astype(str).str.strip()
        if normalize_names:
            events['player'] = events['player'].apply(lambda x: normalize_player_name(x, titlecase=True))

    tracking_p = process_tracking(tracking) if not tracking.empty else pd.DataFrame()
    kpis = compute_kpis(tracking_p, events_df=events if not events.empty else pd.DataFrame()) if not tracking_p.empty else pd.DataFrame()
    zones = compute_zone_distances(tracking_p, nx=nx, ny=ny) if not tracking_p.empty else pd.DataFrame()

    if not kpis.empty and not zones.empty:
        kpis_full = kpis.merge(zones, on="player", how="left")
    elif not kpis.empty:
        kpis_full = kpis.copy()
    elif not zones.empty:
        kpis_full = zones.copy()
    else:
        kpis_full = pd.DataFrame()

    # Ensure all master players appear (placeholder rows)
    players_master_norm = [normalize_player_name(x, titlecase=normalize_names) for x in st.session_state.get('players_master', [])]
    existing_players = set(kpis_full['player'].tolist()) if not kpis_full.empty else set()
    missing_players = [p for p in players_master_norm if p not in existing_players]
    if missing_players:
        placeholder_rows = []
        for p in missing_players:
            placeholder = {
                'player': p,
                'total_distance_m': 0.0,
                'max_speed_m_s': 0.0,
                'avg_speed_m_s': 0.0,
                'hi_distance_m': 0.0,
                'hi_seconds': 0,
                'hi_pct': 0.0,
                'sprint_count_est': 0,
                'accel_count_gt2m_s2': 0,
                'max_accel_m_s2': 0.0,
                'avg_accel_m_s2': 0.0,
                'minutes': 0,
                'passes': 0,
                'shots': 0,
                'xG_sum': 0.0
            }
            if not zones.empty:
                for c in zones.columns:
                    if c.startswith('zone_'):
                        placeholder[c] = 0.0
            placeholder_rows.append(placeholder)
        kpis_full = pd.concat([kpis_full, pd.DataFrame(placeholder_rows)], ignore_index=True, sort=False)

    shots_xg = compute_xg_for_shots(events if not events.empty else pd.DataFrame()) if events is not None else pd.DataFrame()
    if not shots_xg.empty and not kpis_full.empty:
        xg_agg = shots_xg.groupby('player')['xG'].sum().reset_index().rename(columns={'xG': 'xG_sum'})
        kpis_full = kpis_full.merge(xg_agg, on='player', how='left')
        if 'xG_sum' in kpis_full.columns:
            kpis_full['xG_sum'] = kpis_full['xG_sum'].fillna(0.0)
        else:
            kpis_full['xG_sum'] = 0.0
    else:
        if not kpis_full.empty:
            kpis_full['xG_sum'] = 0.0

    # store
    st.session_state['kpis_full'] = kpis_full.copy() if kpis_full is not None else pd.DataFrame()
    st.session_state['shots_xg'] = shots_xg.copy() if not shots_xg.empty else pd.DataFrame()


def run_full_analysis():
    """Load uploaded files or simulate, then store raw data and recompute KPIs."""
    # decide players to simulate
    players_master = st.session_state.get('players_master', [])
    if not players_master:
        # fallback to default list typed in sidebar
        raw_lines = [p.strip() for p in players_text.splitlines() if p.strip()]
        parsed = []
        for line in raw_lines:
            if ',' in line:
                name, _ = [x.strip() for x in line.split(',', 1)]
                parsed.append(normalize_player_name(name, titlecase=normalize_names))
            else:
                parsed.append(normalize_player_name(line, titlecase=normalize_names))
        players_master = parsed
        st.session_state['players_master'] = players_master

    # load tracking/events if uploaded, else simulate
    tracking = pd.DataFrame()
    events = pd.DataFrame()
    if uploaded_tracking is not None:
        try:
            tracking = pd.read_csv(uploaded_tracking)
            st.info("External tracking CSV loaded.")
        except Exception as e:
            st.error(f"Failed to load tracking CSV: {e}")
            tracking = pd.DataFrame()
    if uploaded_events is not None:
        try:
            events = pd.read_csv(uploaded_events)
        except Exception as e:
            st.warning(f"Failed to load events CSV: {e}. Continuing without events.")
            events = pd.DataFrame()

    if tracking.empty:
        tracking, events_sim = simulate_match_data(
            players=players_master,
            minutes_per_half=minutes_per_half,
            seed=seed,
            start_datetime=start_datetime,
            pass_range=(pass_min, pass_max),
            shot_range=(shot_min, shot_max),
            pass_success_prob=pass_success_prob,
            shot_success_prob=shot_success_prob
        )
        if events.empty:
            events = events_sim
        st.info("Using simulated match data.")

    # normalize names if requested
    if not tracking.empty:
        tracking['player'] = tracking['player'].astype(str).str.strip()
        if normalize_names:
            tracking['player'] = tracking['player'].apply(lambda x: normalize_player_name(x, titlecase=True))
    if not events.empty:
        events['player'] = events['player'].astype(str).str.strip()
        if normalize_names:
            events['player'] = events['player'].apply(lambda x: normalize_player_name(x, titlecase=True))

    # store raw data
    st.session_state['last_tracking'] = tracking.copy()
    st.session_state['last_events'] = events.copy()

    # recompute KPIs from session data
    recompute_kpis_from_session()
    st.success("Analysis generated and stored in session.")


# ---------------------------
# React to generate button
# ---------------------------
if 'generate_clicked' in locals() and generate_clicked:
    run_full_analysis()

# ensure at least initial run
if st.session_state.get('kpis_full', pd.DataFrame()).empty and st.session_state.get('last_tracking', pd.DataFrame()).empty:
    # run once to initialize sample data
    run_full_analysis()

# load session results
kpis_full: pd.DataFrame = st.session_state.get('kpis_full', pd.DataFrame())
tracking: pd.DataFrame = st.session_state.get('last_tracking', pd.DataFrame())
events: pd.DataFrame = st.session_state.get('last_events', pd.DataFrame())
shots_xg: pd.DataFrame = st.session_state.get('shots_xg', pd.DataFrame())

# ---------------------------
# Main UI content (KPI overview + selectors)
# ---------------------------
st.subheader("KPI Overview")
if not kpis_full.empty:
    st.dataframe(kpis_full.sort_values('player').reset_index(drop=True))
else:
    st.info("No KPI data available yet (press Generate / Run Analysis).")

# available players (prefer KPIs list but fallback to master list)
if not kpis_full.empty:
    available_players = sorted(kpis_full['player'].unique().tolist())
else:
    available_players = [normalize_player_name(p, titlecase=normalize_names) for p in st.session_state.get('players_master', [])]

# simple role assignment (first player GK by convention, others OUTFIELD)
roles = {p: ('GK' if idx == 0 else 'OUTFIELD') for idx, p in enumerate(available_players)}

st.sidebar.markdown('---')
st.sidebar.header("Role / Subset Filters")
role_filter = st.sidebar.selectbox("Show", options=['All', 'GK', 'OUTFIELD'])
if role_filter == 'GK':
    players_for_ui = [p for p in available_players if roles.get(p, 'OUTFIELD') == 'GK']
elif role_filter == 'OUTFIELD':
    players_for_ui = [p for p in available_players if roles.get(p, 'OUTFIELD') != 'GK']
else:
    players_for_ui = available_players

if len(players_for_ui) == 0:
    st.warning("No players available for the selected role filter. Showing all players.")
    players_for_ui = available_players

# Player selector (stable in session state)
st.subheader("Player-specific Charts")
if not players_for_ui:
    st.info("No players with data available.")
    selected_player = None
else:
    if st.session_state.get('player_sel') not in players_for_ui:
        st.session_state['player_sel'] = players_for_ui[0]
    player_widget_value = st.selectbox("Select Player", players_for_ui,
                                       index=players_for_ui.index(st.session_state['player_sel']) if st.session_state['player_sel'] in players_for_ui else 0,
                                       key='player_sel_widget')
    # sync back
    if st.session_state.get('player_sel_widget') != st.session_state.get('player_sel'):
        st.session_state['player_sel'] = st.session_state['player_sel_widget']
    selected_player = st.session_state.get('player_sel')

# rename capability (propagates to stored tracking/events)
st.markdown("**Rename selected player** (applies to session data)")
new_name = st.text_input("Rename player to:", value=selected_player if selected_player else "")
if st.button('Apply rename') and selected_player and new_name and new_name != selected_player:
    new_norm = normalize_player_name(new_name, titlecase=normalize_names)
    tdf = st.session_state.get('last_tracking', pd.DataFrame()).copy()
    edf = st.session_state.get('last_events', pd.DataFrame()).copy()
    if not tdf.empty:
        tdf.loc[tdf['player'] == selected_player, 'player'] = new_norm
    if not edf.empty:
        edf.loc[edf['player'] == selected_player, 'player'] = new_norm
    pm = st.session_state.get('players_master', [])
    pm = [new_norm if x == selected_player else x for x in pm]
    st.session_state['players_master'] = pm
    st.session_state['last_tracking'] = tdf
    st.session_state['last_events'] = edf
    st.session_state['player_sel'] = new_norm
    st.success(f"Renamed {selected_player} -> {new_norm}. Recomputing KPIs...")
    recompute_kpis_from_session()

# ---------------------------
# Player-specific plots & tables
# ---------------------------
pdata = pd.DataFrame()
if selected_player and not tracking.empty:
    tp = process_tracking(tracking) if not tracking.empty else pd.DataFrame()
    if not tp.empty:
        pdata = tp[tp['player'] == selected_player]

# Speed plot
fig_speed = go.Figure()
if not pdata.empty:
    fig_speed.add_trace(go.Scatter(x=pdata['timestamp'], y=pdata['speed_m_s'], mode='lines', name='Speed'))
fig_speed.update_layout(title=f"Speed over Time - {selected_player}", xaxis_title="Time", yaxis_title="Speed (m/s)", height=300)
st.plotly_chart(fig_speed, use_container_width=True)

# Cumulative distance
fig_cumdist = go.Figure()
if not pdata.empty:
    fig_cumdist.add_trace(go.Scatter(x=pdata['timestamp'], y=pdata['cumulative_distance_m'], mode='lines', name='Cumulative Distance'))
fig_cumdist.update_layout(title=f"Cumulative Distance - {selected_player}", xaxis_title="Time", yaxis_title="Distance (m)", height=300)
st.plotly_chart(fig_cumdist, use_container_width=True)

# Per-half comparison
st.subheader("Per-half Comparison")
fig_half = go.Figure()
if not pdata.empty:
    for h in [1, 2]:
        ph = pdata[pdata['half'] == h]
        fig_half.add_trace(go.Scatter(x=ph['timestamp'], y=ph['cumulative_distance_m'], mode='lines', name=f'Half {h}'))
fig_half.update_layout(title=f"Cumulative Distance per Half - {selected_player}", xaxis_title="Time", yaxis_title="Distance (m)", height=300)
st.plotly_chart(fig_half, use_container_width=True)

# Position heatmap (player)
st.subheader("Position Heatmap")
try:
    if not pdata.empty and pdata['x'].notna().any() and pdata['y'].notna().any():
        hb = np.histogram2d(pdata['x'], pdata['y'], bins=[30, 30], range=[[0, 105], [0, 68]])
        z = hb[0].T
    else:
        z = np.zeros((30, 30))
except Exception:
    z = np.zeros((30, 30))
fig_h = go.Figure(data=go.Heatmap(z=z, x=np.linspace(0, 105, z.shape[1]), y=np.linspace(0, 68, z.shape[0])))
fig_h.update_layout(title=f"Position Heatmap - {selected_player}", xaxis_title="Pitch X (m)", yaxis_title="Pitch Y (m)", height=450)
st.plotly_chart(fig_h, use_container_width=True)
try:
    heat_png = fig_h.to_image(format="png", width=900, height=500, scale=1)
except Exception:
    heat_png = b""

# Zone distances (player)
st.subheader("Distance per Tactical Zone")
zone_cols = [c for c in kpis_full.columns if c.startswith("zone_")] if not kpis_full.empty else []
if zone_cols:
    sel_row = kpis_full.loc[kpis_full['player'] == selected_player] if not kpis_full.empty else pd.DataFrame()
    if not sel_row.empty:
        zone_vals = sel_row.iloc[0][zone_cols].values
    else:
        zone_vals = np.zeros(len(zone_cols))
    fig_zone = go.Figure(data=[go.Bar(x=list(range(len(zone_vals))), y=zone_vals)])
    fig_zone.update_layout(title=f"Distance per Tactical Zone - {selected_player}", xaxis_title="Zone ID", yaxis_title="Distance (m)", height=350)
    st.plotly_chart(fig_zone, use_container_width=True)
    try:
        zone_png = fig_zone.to_image(format="png", width=900, height=400, scale=1)
    except Exception:
        zone_png = b""
else:
    st.info("No zone distance columns available.")
    zone_png = b""

# Pass network (inferred)
st.subheader("Pass Network (inferred)")
pass_net = infer_pass_network(events, tracking) if (isinstance(events, pd.DataFrame) and not events.empty) and (isinstance(tracking, pd.DataFrame) and not tracking.empty) else pd.DataFrame()
if not pass_net.empty and (not tracking.empty):
    node_pos = tracking.groupby('player')[['x', 'y']].mean().reset_index().set_index('player')
    edge_traces = []
    for _, row in pass_net.iterrows():
        if row['from'] in node_pos.index and row['to'] in node_pos.index:
            x0, y0 = node_pos.loc[row['from'], ['x', 'y']]
            x1, y1 = node_pos.loc[row['to'], ['x', 'y']]
            edge_traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                          line=dict(width=max(1, np.log1p(row['count']) * 2)),
                                          hoverinfo='text', text=f"{row['from']} -> {row['to']}: {row['count']}"))
    fig_net = go.Figure()
    for tr in edge_traces:
        fig_net.add_trace(tr)
    fig_net.add_trace(go.Scatter(x=node_pos['x'], y=node_pos['y'], mode='markers+text', text=node_pos.index, textposition='top center', marker=dict(size=10)))
    fig_net.update_layout(title='Pass Network (inferred)', xaxis=dict(range=[0, 105]), yaxis=dict(range=[0, 68]), height=600)
    st.plotly_chart(fig_net, use_container_width=True)
else:
    st.info('Not enough pass data to generate a network.')

# Shots & xG table & scatter
if not shots_xg.empty:
    st.subheader('Shots & xG')
    st.dataframe(shots_xg[['timestamp', 'player', 'x', 'y', 'success', 'dist_to_goal', 'xG']].sort_values('timestamp').reset_index(drop=True))
    try:
        fig_shots = px.scatter(shots_xg, x='x', y='y', size='xG', color='xG', hover_data=['player', 'xG', 'success'])
        fig_shots.update_layout(title='Shots (size/color ~ xG)', xaxis=dict(range=[0, 105]), yaxis=dict(range=[0, 68]), height=500)
        st.plotly_chart(fig_shots, use_container_width=True)
    except Exception:
        pass

# Animated movement (if enabled)
if enable_animation:
    st.subheader('Animated Movement')
    anim_players = st.multiselect('Players to animate (leave empty to animate all)', options=players_for_ui, default=[selected_player] if selected_player in players_for_ui else players_for_ui[:5])
    if len(anim_players) == 0:
        anim_players = players_for_ui[:5]
    fig_anim = create_animation_fig(tracking, players_to_show=anim_players, max_frames=max_anim_frames) if not tracking.empty else None
    if fig_anim is not None:
        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.info('Not enough data to animate.')


# ---------------------------
# Combined team movement (overlay trajectories + heatmap + export)
# ---------------------------
st.subheader("Combined Team Movement")
tracking_all = st.session_state.get('last_tracking', pd.DataFrame())

if tracking_all is None or tracking_all.empty:
    st.info("No tracking available to show combined movement. Run simulation or upload tracking CSV.")
else:
    with st.expander("Combined movement options", expanded=True):
        downsample_rate_all = st.number_input("Downsample step (keep 1 every N samples)", min_value=1, max_value=60, value=5, key="ds_all")
        max_players_to_plot_all = st.number_input("Max players to show trajectories (for clarity)", min_value=1, max_value=50, value=20, key="maxp_all")
        show_trajectories_all = st.checkbox("Show overlay trajectories (lines)", value=True, key="show_traj_all")
        show_heatmap_all = st.checkbox("Show combined heatmap (density)", value=True, key="show_heat_all")
        animate_all = st.checkbox("Show animated movement (all players)", value=False, key="animate_all_ctrl")
        facet_by_half_all = st.checkbox("Split visuals by half", value=False, key="facet_half_all")

    # normalize & prepare
    tracking_all['player'] = tracking_all['player'].astype(str)
    halves_all = sorted(tracking_all['half'].unique().tolist()) if facet_by_half_all else [None]

    overlay_figs = []
    heatmap_figs = []
    fig_labels = []
    color_palette = px.colors.qualitative.Dark24

    for half in halves_all:
        if half is None:
            df = tracking_all.copy()
            suffix = ""
        else:
            df = tracking_all[tracking_all['half'] == half].copy()
            suffix = f" - Half {half}"

        if downsample_rate_all > 1:
            df = df.sort_values(['player', 'timestamp']).groupby('player').apply(lambda g: g.iloc[::downsample_rate_all]).reset_index(drop=True)

        if show_trajectories_all:
            players_to_plot = sorted(df['player'].unique().tolist())[:max_players_to_plot_all]
            traj_fig = go.Figure()
            for i, p in enumerate(players_to_plot):
                gp = df[df['player'] == p]
                if gp.empty:
                    continue
                # faint line trail
                traj_fig.add_trace(go.Scatter(x=gp['x'], y=gp['y'], mode='lines', name=str(p),
                                             line=dict(width=2), opacity=0.35,
                                             hoverinfo='text', text=[f"{p} | {t}" for t in gp['timestamp']]))
                traj_fig.add_trace(go.Scatter(x=gp['x'], y=gp['y'], mode='markers', showlegend=False,
                                             marker=dict(size=4, opacity=0.45)))
            traj_fig.update_layout(title=f"Overlay trajectories (downsample={downsample_rate_all}){suffix}",
                                   xaxis=dict(range=[0, 105], title='Pitch X (m)'),
                                   yaxis=dict(range=[0, 68], title='Pitch Y (m)'),
                                   height=600, legend=dict(itemsizing='constant'))
            st.plotly_chart(traj_fig, use_container_width=True)
            overlay_figs.append(traj_fig)
            fig_labels.append(f"overlay{('_half' + str(half)) if half else '_all'}")

        if show_heatmap_all:
            try:
                hb = np.histogram2d(df['x'], df['y'], bins=[50, 50], range=[[0, 105], [0, 68]])
                z = hb[0].T
            except Exception:
                z = np.zeros((50, 50))
            heat_fig = go.Figure(data=go.Heatmap(z=z, x=np.linspace(0, 105, z.shape[1]), y=np.linspace(0, 68, z.shape[0])))
            heat_fig.update_layout(title=f"Combined position heatmap (all players){suffix}", xaxis=dict(range=[0, 105]), yaxis=dict(range=[0, 68]), height=500)
            st.plotly_chart(heat_fig, use_container_width=True)
            heatmap_figs.append(heat_fig)
            fig_labels.append(f"heatmap{('_half' + str(half)) if half else '_all'}")

    if animate_all:
        all_players_list = sorted(tracking_all['player'].unique().tolist())
        anim_players = st.multiselect("Players to animate (limit for performance)", options=all_players_list, default=all_players_list[:10])
        if not anim_players:
            anim_players = all_players_list[:10]
        anim_fig = create_animation_fig(tracking_all, players_to_show=anim_players, max_frames=min(max_anim_frames, 300))
        if anim_fig is not None:
            st.plotly_chart(anim_fig, use_container_width=True)
        else:
            st.info("Not enough data to animate.")

    # Export combined figures into a ZIP
    st.markdown("---")
    st.markdown("Export combined visuals")
    export_label = st.text_input("ZIP filename (without extension)", value=f"combined_movement_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="zip_name")
    if st.button("Export Combined PNGs to ZIP"):
        to_zip: Dict[str, bytes] = {}
        try:
            for fig, lbl in zip(overlay_figs, [l for l in fig_labels if l.startswith('overlay')]):
                try:
                    png_bytes = fig.to_image(format='png', width=1200, height=800, scale=1)
                except Exception:
                    png_bytes = fig.to_image(format='png', width=900, height=600, scale=1)
                to_zip[f"{lbl}.png"] = png_bytes
            for fig, lbl in zip(heatmap_figs, [l for l in fig_labels if l.startswith('heatmap')]):
                try:
                    png_bytes = fig.to_image(format='png', width=1200, height=800, scale=1)
                except Exception:
                    png_bytes = fig.to_image(format='png', width=900, height=600, scale=1)
                to_zip[f"{lbl}.png"] = png_bytes
            if not kpis_full.empty:
                to_zip['player_kpis_enhanced.csv'] = kpis_full.to_csv(index=False).encode('utf-8')
            else:
                to_zip['player_kpis_enhanced.csv'] = b"player,placeholder\n"
            if not to_zip:
                st.warning("No figures were generated to export.")
            else:
                zip_bytes = make_zip_bytes(to_zip)
                st.download_button("Download combined visuals ZIP", data=zip_bytes, file_name=f"{export_label}.zip", mime="application/zip")
                st.success(f"Prepared {len(to_zip)} files for download in {export_label}.zip")
        except Exception as e:
            st.error(f"Export failed: {e}")


# ---------------------------
# Final downloads area (KPI CSV, PDF, ZIP)
# ---------------------------
st.markdown('---')
st.subheader('Downloads')
csv_bytes = kpis_full.to_csv(index=False).encode('utf-8') if not kpis_full.empty else b''
tracking_csv = tracking.to_csv(index=False).encode('utf-8') if not tracking.empty else b''
events_csv = events.to_csv(index=False).encode('utf-8') if isinstance(events, pd.DataFrame) and not events.empty else b''

metadata = {
    'generated_on': datetime.now().isoformat(),
    'start_datetime': start_datetime.isoformat(),
    'minutes_per_half': minutes_per_half,
    'seed': int(seed),
    'players': st.session_state.get('players_master', []),
    'roles': roles,
    'pass_range': [int(pass_min), int(pass_max)],
    'shot_range': [int(shot_min), int(shot_max)],
    'pass_success_prob': float(pass_success_prob),
    'shot_success_prob': float(shot_success_prob),
    'zones': {'nx': int(nx), 'ny': int(ny)},
    'filters': {'role_filter': role_filter}
}

try:
    pdf_bytes = create_one_page_pdf_bytes(kpis_full if not kpis_full.empty else pd.DataFrame(), heat_png if 'heat_png' in locals() else b"", zone_png if 'zone_png' in locals() else b"")
except Exception:
    pdf_bytes = create_one_page_pdf_bytes(kpis_full if not kpis_full.empty else pd.DataFrame(), b"", b"")

files_for_zip = {
    'player_kpis_enhanced.csv': csv_bytes,
    'VfL_SportsData_OnePage_Report.pdf': pdf_bytes,
    'tracking.csv': tracking_csv,
    'events.csv': events_csv,
    'metadata.json': json.dumps(metadata, indent=2).encode('utf-8')
}

zip_bytes = make_zip_bytes(files_for_zip)

st.download_button('Download KPI CSV', data=csv_bytes, file_name='player_kpis_enhanced.csv', mime='text/csv')
st.download_button('Download One-page PDF', data=pdf_bytes, file_name='VfL_SportsData_OnePage_Report.pdf', mime='application/pdf')
st.download_button('Download ZIP (all artifacts)', data=zip_bytes, file_name='sports_demo_final_package.zip', mime='application/zip')

st.success('Interface ready — use Sync players, Generate, Rename, or Export to save combined visuals.')
