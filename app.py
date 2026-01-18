#!/usr/bin/env python
# Economic activity and emissions (interactive report)
# PCA + KMeans with brushing & linking (box select) + click-to-focus (single point)

import os

# IMPORTANT (Windows stability): set thread vars BEFORE sklearn imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table


# -----------------------------
# Settings
# -----------------------------
DATA_PATH = "owid-co2-data.csv"          # must be in same folder as app.py
YEAR_MIN, YEAR_MAX = 1990, 2020
YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))
DEFAULT_YEAR = 2020

K = 4
FEATURES = ["co2_per_capita", "gdp_per_capita", "energy_per_capita"]

METRIC_LABELS = {
    "co2_per_capita": "CO₂ per capita (tons)",
    "gdp_per_capita": "GDP per capita (USD)",
    "energy_per_capita": "Energy per capita (kWh)"
}

# consistent across all charts
CLUSTER_COLORS = {
    "Cluster 1": "#636EFA",  # blue
    "Cluster 2": "#EF553B",  # orange/red
    "Cluster 3": "#00CC96",  # green
    "Cluster 4": "#AB63FA",  # purple
}


# -----------------------------
# Load + prepare
# -----------------------------
df_all = pd.read_csv(DATA_PATH)

# countries only (ISO3 codes)
df_all = df_all[df_all["iso_code"].astype(str).str.len() == 3].copy()
df_all = df_all[df_all["year"].between(YEAR_MIN, YEAR_MAX)].copy()

# GDP per capita (robust)
df_all["gdp"] = df_all["gdp"].replace(0, np.nan)
df_all["population"] = df_all["population"].replace(0, np.nan)
df_all["gdp_per_capita"] = df_all["gdp"] / df_all["population"]

df_all["country"] = df_all["country"].astype(str)


# -----------------------------
# PCA + KMeans for one year (cached)
# -----------------------------
MODEL_CACHE = {}

def compute_model_for_year(year: int):
    df_y = df_all[df_all["year"] == year].copy()
    df_y = df_y.dropna(subset=FEATURES).copy()

    X = StandardScaler().fit_transform(df_y[FEATURES])

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    df_y["PC1"] = pcs[:, 0]
    df_y["PC2"] = pcs[:, 1]

    exp = pca.explained_variance_ratio_ * 100
    pc1_pct = round(float(exp[0]), 1)
    pc2_pct = round(float(exp[1]), 1)
    total_pct = round(float(exp[0] + exp[1]), 1)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    df_y["cluster_id"] = km.fit_predict(df_y[["PC1", "PC2"]])

    # Label clusters as Cluster 1..4
    df_y["cluster"] = (df_y["cluster_id"].astype(int) + 1).astype(str).radd("Cluster ")

    return df_y, pc1_pct, pc2_pct, total_pct

def get_model(year: int):
    if year not in MODEL_CACHE:
        MODEL_CACHE[year] = compute_model_for_year(year)
    return MODEL_CACHE[year]


# -----------------------------
# Dash app
# -----------------------------
app = dash.Dash(__name__)
server = app.server   # helpful for deployment
app.title = "Economic activity and emissions (interactive report)"


app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial"},
    children=[
        html.H1(
            "Economic activity and emissions (interactive report)",
            style={"textAlign": "center", "marginTop": "18px", "marginBottom": "6px"}
        ),
        html.Div(
            "Tip: click one point to focus on a single country, or drag a box to compare several. The charts and table update automatically.",
            style={"textAlign": "center", "color": "#444", "marginBottom": "18px"}
        ),

        # Top row
        html.Div(
            style={"display": "flex", "gap": "18px", "alignItems": "flex-start"},
            children=[
                # Controls
                html.Div(
                    style={
                        "flex": "1",
                        "border": "1px solid #eee",
                        "borderRadius": "12px",
                        "padding": "14px",
                        "backgroundColor": "white"
                    },
                    children=[
                        html.H3("Controls", style={"marginTop": "0px"}),

                        html.Label("Year"),
                        dcc.Dropdown(
                            id="year",
                            options=[{"label": str(y), "value": y} for y in YEARS],
                            value=DEFAULT_YEAR,
                            clearable=False
                        ),

                        html.Br(),

                        html.Label("Metric shown in summaries"),
                        dcc.Dropdown(
                            id="metric",
                            options=[{"label": METRIC_LABELS[k], "value": k} for k in METRIC_LABELS],
                            value="co2_per_capita",
                            clearable=False
                        ),

                        html.Br(),
                        html.Button("Reset selection", id="reset", n_clicks=0),

                        html.Hr(),

                        html.Div(
                            id="selected-text",
                            style={"color": "#333", "whiteSpace": "pre-wrap"}
                        ),
                    ],
                ),

                # PCA plot
                html.Div(
                    style={"flex": "2"},
                    children=[dcc.Graph(id="pca-scatter", style={"height": "460px"})]
                ),
            ],
        ),

        # Middle row
        html.Div(
            style={"display": "flex", "gap": "18px", "marginTop": "10px"},
            children=[
                dcc.Graph(id="time-series", style={"flex": "1", "height": "330px"}),
                dcc.Graph(id="dist-plot", style={"flex": "1", "height": "330px"}),
            ],
        ),

        # Averages
        html.Div(
            style={"marginTop": "10px"},
            children=[dcc.Graph(id="cluster-avg", style={"height": "300px"})],
        ),

        # Table (own scrollbar)
        html.H3("Selected records", style={"marginTop": "8px"}),

        dash_table.DataTable(
            id="data-table",
            page_size=12,
            sort_action="native",
            filter_action="native",
            fixed_rows={"headers": True},
            style_table={
                "height": "360px",
                "overflowY": "auto",
                "overflowX": "auto"
            },
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f7f7f7"},
        ),

        html.Div(
            id="footer",
            style={"marginTop": "12px", "color": "#666", "fontSize": "12px", "textAlign": "center"}
        ),
    ]
)


# -----------------------------
# PCA plot updates by YEAR
# -----------------------------
@app.callback(
    Output("pca-scatter", "figure"),
    Output("footer", "children"),
    Input("year", "value"),
)
def update_pca(year):
    df_y, pc1_pct, pc2_pct, total_pct = get_model(year)

    fig = px.scatter(
        df_y,
        x="PC1",
        y="PC2",
        color="cluster",
        color_discrete_map=CLUSTER_COLORS,
        hover_name="country",
        hover_data={
            "co2_per_capita": ":.3f",
            "gdp_per_capita": ":.0f",
            "energy_per_capita": ":.1f",
            "cluster": False
        },
        title=f"PCA + K-Means (k={K}) for {year}",
    )

    fig.update_layout(
        dragmode="select",          # box select
        clickmode="event+select",   # click select (single point)
        legend_title_text="Cluster",
        xaxis_title=f"PC1 ({pc1_pct}%)",
        yaxis_title=f"PC2 ({pc2_pct}%)",
        margin=dict(l=40, r=10, t=60, b=40),
    )

    footer = (
        f"Explained variance: PC1 {pc1_pct}%, PC2 {pc2_pct}% (total {total_pct}%). "
        f"Countries used in model: {len(df_y)}."
    )
    return fig, footer


# Reset selection clears both box-select and click-select
@app.callback(
    Output("pca-scatter", "selectedData"),
    Output("pca-scatter", "clickData"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_selection(_n):
    return None, None


# -----------------------------
# Brushing & linking (selection -> other views + table)
# -----------------------------
@app.callback(
    Output("time-series", "figure"),
    Output("dist-plot", "figure"),
    Output("cluster-avg", "figure"),
    Output("data-table", "columns"),
    Output("data-table", "data"),
    Output("selected-text", "children"),
    Input("pca-scatter", "selectedData"),
    Input("pca-scatter", "clickData"),
    Input("year", "value"),
    Input("metric", "value"),
)
def update_linked(selectedData, clickData, year, metric):
    df_y, _, _, _ = get_model(year)

    selected_countries = []

    # (1) box selection (multiple)
    if selectedData and selectedData.get("points"):
        selected_countries = sorted({
            p.get("hovertext") for p in selectedData["points"] if p.get("hovertext")
        })

    # (2) click selection (single) if no box selection
    elif clickData and clickData.get("points"):
        ht = clickData["points"][0].get("hovertext")
        if ht:
            selected_countries = [ht]

    # Filter yearly view for distribution/avg/table
    df_y_sel = df_y[df_y["country"].isin(selected_countries)].copy() if selected_countries else df_y.copy()

    # Time-series uses df_all across years
    if selected_countries:
        df_ts = df_all[df_all["country"].isin(selected_countries)].copy()
        title_ts = f"Trend over time (selection): {METRIC_LABELS[metric]}"
    else:
        # default: top 8 in selected year
        top = df_y.sort_values(metric, ascending=False)["country"].head(8).tolist()
        df_ts = df_all[df_all["country"].isin(top)].copy()
        title_ts = f"Trend over time (default: top 8 in {year}): {METRIC_LABELS[metric]}"

    fig_time = px.line(
        df_ts.dropna(subset=[metric]),
        x="year",
        y=metric,
        color="country",
        title=title_ts,
    )
    fig_time.update_layout(margin=dict(l=40, r=10, t=60, b=40), legend_title_text="Country")

    # Distribution by cluster (colored)
    fig_dist = px.box(
        df_y_sel.dropna(subset=[metric]),
        x="cluster",
        y=metric,
        color="cluster",
        color_discrete_map=CLUSTER_COLORS,
        points="outliers",
        title=f"Distribution by cluster ({year}): {METRIC_LABELS[metric]}",
    )
    fig_dist.update_layout(margin=dict(l=40, r=10, t=60, b=40), xaxis_title="Cluster", yaxis_title=METRIC_LABELS[metric])

    # Cluster averages (colored)
    avg = df_y_sel.groupby("cluster", as_index=False)[metric].mean()
    fig_avg = px.bar(
        avg,
        x="cluster",
        y=metric,
        color="cluster",
        color_discrete_map=CLUSTER_COLORS,
        title=f"Cluster averages ({year}): {METRIC_LABELS[metric]}",
    )
    fig_avg.update_layout(margin=dict(l=40, r=10, t=60, b=40), xaxis_title="Cluster", yaxis_title=f"Average {METRIC_LABELS[metric]}")

    # Table
    show_cols = ["country", "year"] + FEATURES + ["cluster"]
    df_table = df_y_sel[show_cols].copy().sort_values(["cluster", metric], ascending=[True, False])

    columns = [{"name": c.replace("_", " ").title(), "id": c} for c in show_cols]
    data = df_table.to_dict("records")

    if selected_countries:
        sel_txt = "Selected countries:\n" + ", ".join(selected_countries)
    else:
        sel_txt = "No selection yet. Click one point (one country) or drag a box (multiple countries)."

    return fig_time, fig_dist, fig_avg, columns, data, sel_txt


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=True)



