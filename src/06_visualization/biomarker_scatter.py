import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys

CHEMO_CSV   = "./Chemotherapy/biomarker_rxns_mapped.csv"
RADIO_CSV   = "./Radiation_Therapy/biomarker_rxns_mapped.csv"
OUTPUT_HTML = "./biomarker_scatter.html"

def load(path, label):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"rxn_id", "log2_fold_change", "rxn_name"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] {label} CSV missing columns: {missing}")
        sys.exit(1)
    df = df[["rxn_id","log2_fold_change","rxn_name"]].copy()
    df["log2_fold_change"] = pd.to_numeric(df["log2_fold_change"], errors="coerce")
    return df.dropna(subset=["log2_fold_change"])

chemo = load(CHEMO_CSV, "Chemo")
radio = load(RADIO_CSV, "Radio")

merged     = chemo.merge(radio, on="rxn_id", suffixes=("_chemo","_radio"))
chemo_only = chemo[~chemo["rxn_id"].isin(merged["rxn_id"])].copy()
radio_only = radio[~radio["rxn_id"].isin(merged["rxn_id"])].copy()
print(f"Shared:{len(merged)}  Chemo-only:{len(chemo_only)}  Radio-only:{len(radio_only)}")

# ── UNIFORM GRID MAPPING ──────────────────────────────────────────────────────
# Every tick occupies exactly 1 unit of display space.
# Real values → display positions (uniform step = 1)

# Y axis ticks (real values) in order low→high:
Y_REAL    = [-30, -7, -5, -3, -1,  0, 0.5, 1.0, 1.5, 2.0]
Y_DISPLAY = [ -4, -3, -2, -1,  0,  1,   2,   3,   4,   5]  # uniform step=1

# X axis ticks (real values) in order low→high:
X_REAL    = [-6, -5, -4, -3, -2, -1,    0, 0.25, 0.5, 0.75, 1.0]
X_DISPLAY = [-5, -4, -3, -2, -1,  0,    1,    2,   3,    4,   5]  # uniform step=1

def to_display(v, real_arr, disp_arr):
    """Map a real value to its uniform display coordinate via linear interpolation."""
    real_arr = np.array(real_arr, dtype=float)
    disp_arr = np.array(disp_arr, dtype=float)
    if v <= real_arr[0]:  return disp_arr[0]
    if v >= real_arr[-1]: return disp_arr[-1]
    for i in range(len(real_arr)-1):
        if real_arr[i] <= v <= real_arr[i+1]:
            t = (v - real_arr[i]) / (real_arr[i+1] - real_arr[i])
            return disp_arr[i] + t * (disp_arr[i+1] - disp_arr[i])
    return disp_arr[-1]

def tx(v): return to_display(v, X_REAL, X_DISPLAY)
def ty(v): return to_display(v, Y_REAL, Y_DISPLAY)

def dir(fc): return "higher in Sensitive" if fc > 0 else "higher in Resistant"

# ── TICK LABELS (bold -30) ────────────────────────────────────────────────────
y_tickvals = Y_DISPLAY
y_ticktext = ["<b>−30</b>" if r==-30 else f"<b>{int(r) if r==int(r) else r}</b>" for r in Y_REAL]

x_tickvals = X_DISPLAY
x_ticktext = [f"<b>{int(r) if r==int(r) else r}</b>" for r in X_REAL]

# axis display limits
xmin, xmax = X_DISPLAY[0]-0.4, X_DISPLAY[-1]+0.4
ymin, ymax = Y_DISPLAY[0]-0.4, Y_DISPLAY[-1]+0.4

# zero-axis display coords
x0d = tx(0)   # display x=0
y0d = ty(0)   # display y=0

# ── HOVER ─────────────────────────────────────────────────────────────────────
def hover_shared(row):
    same = (row.log2_fold_change_chemo>0)==(row.log2_fold_change_radio>0)
    return (f"<b>{row.rxn_id}</b><br>{row.rxn_name_chemo}<br><br>"
            f"Chemo log₂FC: {row.log2_fold_change_chemo:+.3f}  ({dir(row.log2_fold_change_chemo)})<br>"
            f"Radio log₂FC: {row.log2_fold_change_radio:+.3f}  ({dir(row.log2_fold_change_radio)})<br><br>"
            f"<i>{'Same direction in both ✓' if same else 'Opposite directions ✗'}</i>")

def hover_chemo(row):
    return (f"<b>{row.rxn_id}</b><br>{row.rxn_name}<br><br>"
            f"Chemo log₂FC: {row.log2_fold_change:+.3f}  ({dir(row.log2_fold_change)})<br><br>"
            f"<i>Chemo-only biomarker</i>")

def hover_radio(row):
    return (f"<b>{row.rxn_id}</b><br>{row.rxn_name}<br><br>"
            f"Radio log₂FC: {row.log2_fold_change:+.3f}  ({dir(row.log2_fold_change)})<br><br>"
            f"<i>Radio-only biomarker</i>")

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = go.Figure()

shapes = [
    # Q1 both sensitive
    dict(type="rect",xref="x",yref="y",x0=x0d,x1=xmax,y0=y0d,y1=ymax,
         fillcolor="rgba(29,158,117,0.08)",line_width=0,layer="below"),
    # Q3 both resistant
    dict(type="rect",xref="x",yref="y",x0=xmin,x1=x0d,y0=ymin,y1=y0d,
         fillcolor="rgba(220,55,55,0.07)",line_width=0,layer="below"),
    # Q2 radio-sensitive/chemo-resistant
    dict(type="rect",xref="x",yref="y",x0=xmin,x1=x0d,y0=y0d,y1=ymax,
         fillcolor="rgba(239,159,39,0.04)",line_width=0,layer="below"),
    # Q4 chemo-sensitive/radio-resistant
    dict(type="rect",xref="x",yref="y",x0=x0d,x1=xmax,y0=ymin,y1=y0d,
         fillcolor="rgba(55,138,221,0.04)",line_width=0,layer="below"),
    # zero lines
    dict(type="line",xref="x",yref="y",x0=xmin,x1=xmax,y0=y0d,y1=y0d,
         line=dict(color="rgba(120,120,120,0.45)",width=1,dash="dash")),
    dict(type="line",xref="x",yref="y",x0=x0d,x1=x0d,y0=ymin,y1=ymax,
         line=dict(color="rgba(120,120,120,0.45)",width=1,dash="dash")),
    # -30 highlight line (dotted red)
    dict(type="line",xref="x",yref="y",x0=xmin,x1=xmax,
         y0=ty(-30),y1=ty(-30),
         line=dict(color="rgba(210,60,60,0.4)",width=1,dash="dot")),
]

annotations = [
    dict(x=0.98,y=0.98,xref="paper",yref="paper",text="Both Sensitive ↗",
         showarrow=False,font=dict(size=11,color="rgba(20,140,100,0.75)"),
         xanchor="right",yanchor="top"),
    dict(x=0.02,y=0.02,xref="paper",yref="paper",text="↙ Both Resistant",
         showarrow=False,font=dict(size=11,color="rgba(200,50,50,0.75)"),
         xanchor="left",yanchor="bottom"),
    dict(x=0.02,y=0.98,xref="paper",yref="paper",
         text="Radio-Sensitive / Chemo-Resistant",
         showarrow=False,font=dict(size=11,color="rgba(180,120,20,0.7)"),
         xanchor="left",yanchor="top"),
    dict(x=0.98,y=0.02,xref="paper",yref="paper",
         text="Chemo-Sensitive / Radio-Resistant",
         showarrow=False,font=dict(size=11,color="rgba(40,110,200,0.7)"),
         xanchor="right",yanchor="bottom"),
]

# Traces — map real values through uniform transform
fig.add_trace(go.Scatter(
    x=[tx(r.log2_fold_change_chemo) for _,r in merged.iterrows()],
    y=[ty(r.log2_fold_change_radio)  for _,r in merged.iterrows()],
    mode="markers", name=f"Both therapies (n={len(merged)})",
    marker=dict(color="rgba(123,104,238,0.88)",size=10,
                line=dict(color="#4A3FCC",width=1.2)),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=[hover_shared(r) for _,r in merged.iterrows()],
))

fig.add_trace(go.Scatter(
    x=[tx(r.log2_fold_change) for _,r in chemo_only.iterrows()],
    y=[ty(0)]*len(chemo_only),
    mode="markers", name=f"Chemo-only (n={len(chemo_only)})",
    marker=dict(color="rgba(26,188,176,0.88)",size=9,symbol="triangle-up",
                line=dict(color="#0E8A82",width=1)),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=[hover_chemo(r) for _,r in chemo_only.iterrows()],
))

fig.add_trace(go.Scatter(
    x=[tx(0)]*len(radio_only),
    y=[ty(r.log2_fold_change) for _,r in radio_only.iterrows()],
    mode="markers", name=f"Radio-only (n={len(radio_only)})",
    marker=dict(color="rgba(245,166,35,0.88)",size=9,symbol="square",
                line=dict(color="#C47F00",width=1)),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=[hover_radio(r) for _,r in radio_only.iterrows()],
))

fig.update_layout(
    shapes=shapes, annotations=annotations,
    xaxis=dict(
        range=[xmin,xmax], zeroline=False,
        gridcolor="rgba(150,150,150,0.13)",
        tickvals=x_tickvals, ticktext=x_ticktext,
        tickfont=dict(size=12,family="Arial Black"),
        title=dict(text="Chemo log₂FC (Sensitive / Resistant)",
                   font=dict(size=13)),
        mirror=True, showline=True, linecolor="rgba(100,100,100,0.4)",
    ),
    yaxis=dict(
        range=[ymin,ymax], zeroline=False,
        gridcolor="rgba(150,150,150,0.13)",
        tickvals=y_tickvals, ticktext=y_ticktext,
        tickfont=dict(size=12,family="Arial Black"),
        title=dict(text="Radio log₂FC (Sensitive / Resistant)",
                   font=dict(size=13)),
        mirror=True, showline=True, linecolor="rgba(100,100,100,0.4)",
    ),
    legend=dict(orientation="h",yanchor="bottom",y=1.02,
                xanchor="center",x=0.5,font=dict(size=12),itemsizing="constant"),
    hoverlabel=dict(bgcolor="rgba(20,20,20,0.93)",
                    font=dict(color="white",size=12),
                    bordercolor="rgba(255,255,255,0.15)"),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=80,r=80,t=80,b=80),
    width=860, height=680,
)

html = fig.to_html(include_plotlyjs="cdn", full_html=True)
html = html.replace("<body>",
    "<body style='display:flex;justify-content:center;padding:20px;background:#fff;'>")
with open(OUTPUT_HTML,"w") as f:
    f.write(html)
print(f"Saved → {OUTPUT_HTML}")
