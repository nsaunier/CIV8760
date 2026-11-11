"""Interactive k-means clustering demo (Dash + Plotly).

A companion to knn_demo.py: a 2D set of unlabelled points is clustered with
k-means (Lloyd's algorithm), one step at a time. Each press of "Step" runs
the next half-iteration, alternating between

  1. assignment: every point is assigned to its nearest centroid, and
  2. update: every centroid moves to the mean of the points assigned to it,

until the assignments stop changing. The background shows the region of the
plane closest to each centroid (its Voronoi cell) and the lines trace the
path of each centroid. Click on the plot to add a point.

Run with:  python kmeans_demo.py   then open http://127.0.0.1:8050
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

# Plot domain and Voronoi-region grid resolution
LIM = 1.0
GRID_RES = 160
MAX_K = 8
MAX_RUN_STEPS = 200

# Saturated colours for points and centroids, pale versions for the regions
POINT_COLORS = ["#e3342f", "#38a169", "#3182ce", "#805ad5", "#dd8b0b", "#d53f8c", "#319795", "#744210"]
REGION_COLORS = ["#f9c9c7", "#c3e8cf", "#c3dbf2", "#dccff4", "#f8e0b5", "#f6c6df", "#bfe3e1", "#e2d2c0"]
UNASSIGNED_COLOR = "#a0aec0"

DATASETS = {
    "blobs": "Gaussian blobs",
    "elongated": "Elongated blobs",
    "uneven": "Blobs of uneven size",
    "rings": "Concentric rings",
    "uniform": "Uniform random",
}

INITS = {
    "points": "random data points (Forgy)",
    "uniform": "uniformly random positions",
    "kmeans++": "k-means++",
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_dataset(kind, n_groups, n_points, seed):
    """Unlabelled points; n_groups is the number of underlying groups."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        X = rng.uniform(-LIM, LIM, size=(n_points, 2))
    elif kind == "rings":
        group = rng.integers(n_groups, size=n_points)
        r = 0.15 + 0.75 * group / max(n_groups - 1, 1) + rng.normal(scale=0.04, size=n_points)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        X = np.c_[r * np.cos(theta), r * np.sin(theta)]
    else:
        centers = rng.uniform(-0.65, 0.65, size=(n_groups, 2))
        if kind == "uneven":
            sizes = rng.dirichlet(np.full(n_groups, 0.7))
            scales = rng.uniform(0.05, 0.3, size=n_groups)
        else:
            sizes = np.full(n_groups, 1 / n_groups)
            scales = np.full(n_groups, 0.15)
        group = rng.choice(n_groups, size=n_points, p=sizes)
        X = centers[group] + rng.normal(size=(n_points, 2)) * scales[group, None]
        if kind == "elongated":
            # stretch each blob along a random direction
            angle = rng.uniform(0, np.pi, n_groups)[group]
            local = X - centers[group]
            u = local[:, 0] * 2.2
            v = local[:, 1] * 0.35
            X = centers[group] + np.c_[u * np.cos(angle) - v * np.sin(angle), u * np.sin(angle) + v * np.cos(angle)]
    X = np.clip(X, -LIM * 0.98, LIM * 0.98)
    return {"x": X[:, 0].tolist(), "y": X[:, 1].tolist()}


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------
def sq_distances(Q, C):
    """Squared Euclidean distances between points Q (m, 2) and centroids C (k, 2)."""
    return ((Q[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def init_centroids(X, k, method, seed):
    rng = np.random.default_rng(seed)
    if method == "uniform" or len(X) == 0:
        return rng.uniform(-LIM * 0.9, LIM * 0.9, size=(k, 2))
    if method == "points" or len(X) < k:
        return X[rng.choice(len(X), size=k, replace=len(X) < k)]
    # k-means++: each new centroid is a data point drawn with probability
    # proportional to its squared distance to the closest centroid so far
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = sq_distances(X, np.asarray(C)).min(axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        C.append(X[rng.choice(len(X), p=p)])
    return np.asarray(C)


def new_state(X, k, method, seed):
    C = init_centroids(X, k, method, seed)
    return {
        "centroids": C.tolist(),
        "history": [C.tolist()],          # centroid positions after each update
        "assign": [-1] * len(X),          # -1: not assigned yet
        "next": "assign",                 # which step runs next
        "iteration": 0,
        "converged": False,
        "message": "Centroids initialised. Next step: assign each point to its nearest centroid.",
    }


def step(state, X):
    """Run the next half-iteration of Lloyd's algorithm. Returns a new state."""
    state = dict(state)
    C = np.asarray(state["centroids"])
    old = np.asarray(state["assign"], dtype=int)
    if state["next"] == "assign":
        assign = sq_distances(X, C).argmin(axis=1) if len(X) else np.empty(0, dtype=int)
        n_changed = int((assign != old).sum())
        state["assign"] = assign.tolist()
        state["next"] = "update"
        if n_changed == 0 and len(X):
            state["converged"] = True
            state["message"] = "No point changed cluster: k-means has converged."
        else:
            state["message"] = (f"Assignment: {n_changed} point(s) changed cluster. "
                                "Next step: move each centroid to the mean of its points.")
    else:
        new_C = C.copy()
        empty = []
        for j in range(len(C)):
            members = X[old == j]
            if len(members):
                new_C[j] = members.mean(axis=0)
            else:
                empty.append(j)  # an empty cluster keeps its centroid
        shift = float(np.sqrt(((new_C - C) ** 2).sum(axis=1)).max()) if len(C) else 0.0
        state["centroids"] = new_C.tolist()
        state["history"] = state["history"] + [new_C.tolist()]
        state["iteration"] += 1
        state["next"] = "assign"
        state["message"] = (f"Update (iteration {state['iteration']}): centroids moved by up to {shift:.3f}."
                            + (f" Cluster(s) {', '.join(str(j + 1) for j in empty)} empty: centroid left in place."
                               if empty else "")
                            + " Next step: reassign each point to its nearest centroid.")
    return state


def inertia(X, state):
    """Sum of squared distances of the points to their assigned centroid."""
    assign = np.asarray(state["assign"], dtype=int)
    m = assign >= 0
    if not m.any():
        return None
    C = np.asarray(state["centroids"])
    return float(((X[m] - C[assign[m]]) ** 2).sum())


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(X, state, show_regions, show_paths):
    C = np.asarray(state["centroids"])
    k = len(C)
    assign = np.asarray(state["assign"], dtype=int)
    fig = go.Figure()

    # Voronoi cells of the current centroids, as a heatmap with a discrete colorscale
    g = np.linspace(-LIM, LIM, GRID_RES)
    if show_regions:
        gx, gy = np.meshgrid(g, g)
        Z = sq_distances(np.c_[gx.ravel(), gy.ravel()], C).argmin(axis=1).reshape(GRID_RES, GRID_RES)
        colorscale = []
        for i, c in enumerate(REGION_COLORS[:k]):
            colorscale += [[i / k, c], [(i + 1) / k, c]]
        zmin, zmax = -0.5, k - 0.5
    else:
        Z, colorscale, zmin, zmax = np.zeros((GRID_RES, GRID_RES)), [[0, "white"], [1, "white"]], 0, 1
    # always draw the heatmap so that clicks anywhere on the plot are captured
    fig.add_trace(go.Heatmap(x=g, y=g, z=Z, zmin=zmin, zmax=zmax, colorscale=colorscale,
                             showscale=False, hoverinfo="none", name="regions"))

    # Data points, coloured by their current assignment
    groups = [(-1, "unassigned", UNASSIGNED_COLOR)] + [(j, f"cluster {j + 1}", POINT_COLORS[j]) for j in range(k)]
    for j, name, color in groups:
        m = assign == j
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=X[m, 0], y=X[m, 1], mode="markers", name=name, showlegend=False,
            marker=dict(size=8, color=color, line=dict(width=0.8, color="black")),
            hovertemplate=f"{name}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
        ))

    # Paths followed by the centroids across update steps
    if show_paths and len(state["history"]) > 1:
        H = np.asarray(state["history"])  # (steps, k, 2)
        for j in range(k):
            fig.add_trace(go.Scatter(
                x=H[:, j, 0], y=H[:, j, 1], mode="lines+markers", hoverinfo="skip", showlegend=False,
                line=dict(color=POINT_COLORS[j], width=2),
                marker=dict(size=5, color=POINT_COLORS[j], line=dict(width=0.5, color="black")),
            ))

    # Centroids
    fig.add_trace(go.Scatter(
        x=C[:, 0], y=C[:, 1], mode="markers", name="centroids", showlegend=False,
        marker=dict(symbol="x", size=18, color=POINT_COLORS[:k], line=dict(width=2, color="black")),
        customdata=np.arange(1, k + 1),
        hovertemplate="centroid %{customdata}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-LIM, LIM], constrain="domain", showgrid=False, zeroline=False),
        yaxis=dict(range=[-LIM, LIM], scaleanchor="x", showgrid=False, zeroline=False),
        plot_bgcolor="white", hovermode="closest", hoverdistance=6,
        uirevision="keep",  # preserve zoom/pan across updates
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def control(label, component):
    return html.Div([html.Label(label, style={"fontWeight": 600}), component], style={"marginBottom": "18px"})


BUTTON_ROW = {"display": "flex", "flexWrap": "wrap", "gap": "8px", "marginBottom": "18px"}

app = Dash(__name__, title="k-means demo")
app.layout = html.Div(
    style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "padding": "16px",
           "fontFamily": "system-ui, sans-serif", "maxWidth": "1200px", "margin": "auto"},
    children=[
        dcc.Store(id="data"),
        dcc.Store(id="state"),
        dcc.Store(id="seeds", data={"data": 0, "init": 0}),
        dcc.Interval(id="timer", interval=700, disabled=True),
        html.Div(style={"flex": "1 1 300px", "maxWidth": "340px"}, children=[
            html.H2("k-means clustering", style={"marginTop": 0}),
            control("Number of clusters k", dcc.Slider(
                id="k", min=1, max=MAX_K, step=1, value=3)),
            control("Initialisation", dcc.Dropdown(
                id="init", value="points", clearable=False,
                options=[{"label": v, "value": k} for k, v in INITS.items()])),
            html.Div(style=BUTTON_ROW, children=[
                html.Button("Step: assign points", id="step", n_clicks=0,
                            style={"fontWeight": 600}),
                html.Button("Run to convergence", id="run", n_clicks=0),
                html.Button("Play", id="play", n_clicks=0),
                html.Button("Re-initialise centroids", id="reinit", n_clicks=0),
            ]),
            control("Display", dcc.Checklist(
                id="display", value=["regions", "paths"],
                options=[{"label": " regions closest to each centroid", "value": "regions"},
                         {"label": " centroid paths", "value": "paths"}])),
            html.Hr(),
            control("Dataset", dcc.Dropdown(
                id="dataset", value="blobs", clearable=False,
                options=[{"label": v, "value": k} for k, v in DATASETS.items()])),
            control("Number of underlying groups", dcc.Slider(id="n-groups", min=1, max=MAX_K, step=1, value=4)),
            control("Number of points", dcc.Slider(
                id="n-points", min=20, max=500, step=10, value=150,
                marks={i: str(i) for i in (20, 100, 200, 300, 400, 500)})),
            html.Div(style=BUTTON_ROW, children=[
                html.Button("Randomize points", id="randomize", n_clicks=0),
                html.Button("Clear all points", id="clear", n_clicks=0),
            ]),
            html.P("Click on the plot to add a point.", style={"color": "#666", "fontSize": "0.9em"}),
            html.Hr(),
            html.Div(id="stats"),
        ]),
        html.Div(style={"flex": "3 1 500px", "minWidth": "320px"}, children=[
            dcc.Graph(id="graph", style={"height": "75vh", "minHeight": "420px"},
                      config={"displaylogo": False}),
            html.Div(id="message", style={"marginTop": "6px", "color": "#444"}),
        ]),
    ],
)


def as_array(data):
    return np.c_[data["x"], data["y"]] if data and data["x"] else np.empty((0, 2))


@app.callback(
    Output("data", "data"),
    Output("state", "data"),
    Output("seeds", "data"),
    Output("timer", "disabled"),
    Input("dataset", "value"),
    Input("n-groups", "value"),
    Input("n-points", "value"),
    Input("randomize", "n_clicks"),
    Input("clear", "n_clicks"),
    Input("graph", "clickData"),
    Input("k", "value"),
    Input("init", "value"),
    Input("reinit", "n_clicks"),
    Input("step", "n_clicks"),
    Input("run", "n_clicks"),
    Input("play", "n_clicks"),
    Input("timer", "n_intervals"),
    State("data", "data"),
    State("state", "data"),
    State("seeds", "data"),
    State("timer", "disabled"),
)
def update_model(dataset, n_groups, n_points, _rand, _clear, click, k, init, _reinit, _step, _run, _play,
                 _tick, data, state, seeds, paused):
    trigger = ctx.triggered_id
    seeds = dict(seeds)

    # Changes to the data
    if trigger == "graph" and click and data is not None and state is not None:
        pt = click["points"][0]
        data = {"x": data["x"] + [float(pt["x"])], "y": data["y"] + [float(pt["y"])]}
        # keep the centroids; the new point is unassigned until the next assignment step
        state = dict(state, assign=state["assign"] + [-1], next="assign", converged=False,
                     message="Point added. Next step: assign each point to its nearest centroid.")
        return data, state, seeds, paused
    if trigger == "clear":
        data = {"x": [], "y": []}
    elif trigger in (None, "dataset", "n-groups", "n-points", "randomize") or data is None:
        if trigger == "randomize":
            seeds["data"] += 1
        data = make_dataset(dataset, n_groups, n_points, seeds["data"])
    X = as_array(data)

    # Steps of the algorithm
    if trigger in ("step", "run", "timer") and state is not None:
        if state["converged"]:
            return data, state, seeds, True
        state = step(state, X)
        if trigger == "run":
            for _ in range(MAX_RUN_STEPS):
                if state["converged"]:
                    break
                state = step(state, X)
        return data, state, seeds, paused or state["converged"]
    if trigger == "play":
        if state is not None and state["converged"]:
            return data, state, seeds, True
        return data, state, seeds, not paused

    # Anything else (new data, k, initialisation) restarts from new centroids
    if trigger == "reinit":
        seeds["init"] += 1
    return data, new_state(X, k, init, seeds["init"]), seeds, True


@app.callback(
    Output("graph", "figure"),
    Output("message", "children"),
    Output("stats", "children"),
    Output("step", "children"),
    Output("step", "disabled"),
    Output("run", "disabled"),
    Output("play", "children"),
    Output("play", "disabled"),
    Input("state", "data"),
    Input("display", "value"),
    Input("timer", "disabled"),
    State("data", "data"),
)
def render(state, display, paused, data):
    if state is None:
        state = new_state(np.empty((0, 2)), 1, "uniform", 0)
    X = as_array(data)
    fig = build_figure(X, state, "regions" in display, "paths" in display)
    J = inertia(X, state)
    stats = [
        html.Div(f"{len(X)} points, k = {len(state['centroids'])}"),
        html.Div(f"Iterations (centroid updates): {state['iteration']}"),
        html.Div("Inertia (sum of squared distances to the assigned centroid): "
                 + (f"{J:.3f}" if J is not None else "–")),
        html.Div("Converged" if state["converged"] else "Not converged yet",
                 style={"fontWeight": 600, "color": "#38a169" if state["converged"] else "#dd8b0b"}),
        html.P(
            "Each assignment step and each update step can only decrease the inertia, so k-means always "
            "converges, but only to a local minimum: try re-initialising the centroids or k-means++. "
            "k-means also assumes compact, round clusters of similar size; compare with the elongated, "
            "uneven and ring datasets.",
            style={"color": "#666", "fontSize": "0.9em"}),
    ]
    step_label = "Step: assign points" if state["next"] == "assign" else "Step: update centroids"
    done = state["converged"] or not len(X)
    return (fig, state["message"], stats, step_label, done, done,
            "Play" if paused else "Pause", done)


if __name__ == "__main__":
    app.run(debug=True)
