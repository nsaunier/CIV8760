"""Interactive k-Nearest Neighbours demo (Dash + Plotly).

Inspired by the Stanford CS231n k-NN demo: a 2D set of labelled points is
shown on top of the decision regions of a k-NN classifier. You can change k,
the distance metric, the number of classes and points, and click on the plot
to add or remove points, or to query a location and see its k neighbours.

Run with:  python knn_demo.py   then open http://127.0.0.1:8050
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

# Plot domain and decision-region grid resolution
LIM = 1.0
GRID_RES = 160

# Saturated colours for the points, pale versions for the regions
POINT_COLORS = ["#e3342f", "#38a169", "#3182ce", "#805ad5", "#dd8b0b"]
REGION_COLORS = ["#f9c9c7", "#c3e8cf", "#c3dbf2", "#dccff4", "#f8e0b5"]
TIE_COLOR = "#ffffff"
CLASS_NAMES = ["red", "green", "blue", "purple", "orange"]

DATASETS = {
    "blobs": "Gaussian blobs",
    "random": "Uniform random (noisy labels)",
    "circles": "Concentric rings",
    "spiral": "Spirals",
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def make_dataset(kind, n_classes, n_per_class, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        if kind == "blobs":
            center = rng.uniform(-0.6, 0.6, size=2)
            pts = center + rng.normal(scale=0.22, size=(n_per_class, 2))
        elif kind == "random":
            pts = rng.uniform(-LIM, LIM, size=(n_per_class, 2))
        elif kind == "circles":
            r = 0.2 + 0.7 * c / max(n_classes - 1, 1)
            theta = rng.uniform(0, 2 * np.pi, n_per_class)
            r = r + rng.normal(scale=0.06, size=n_per_class)
            pts = np.c_[r * np.cos(theta), r * np.sin(theta)]
        elif kind == "spiral":
            t = np.linspace(0.05, 1, n_per_class)
            theta = t * 4 + c * 2 * np.pi / n_classes + rng.normal(scale=0.35, size=n_per_class)
            pts = np.c_[0.9 * t * np.cos(theta), 0.9 * t * np.sin(theta)]
        else:
            raise ValueError(f"unknown dataset {kind!r}")
        xs.append(pts)
        ys.append(np.full(n_per_class, c))
    X = np.clip(np.vstack(xs), -LIM * 0.98, LIM * 0.98)
    return {"x": X[:, 0].tolist(), "y": X[:, 1].tolist(), "label": np.concatenate(ys).tolist()}


# ---------------------------------------------------------------------------
# k-NN
# ---------------------------------------------------------------------------
def distances(Q, X, metric):
    """Pairwise distances between queries Q (m, 2) and training points X (n, 2)."""
    dx = Q[:, 0, None] - X[None, :, 0]
    dy = Q[:, 1, None] - X[None, :, 1]
    if metric == "L1":
        return np.abs(dx) + np.abs(dy)
    return np.sqrt(dx * dx + dy * dy)


def knn_vote(D, labels, k, n_classes, ties):
    """Majority vote among the k nearest training points for each row of D.

    Returns (predictions, sorted neighbour indices). A prediction of -1 means
    the vote was tied and ties == "undecided".
    """
    m, n = D.shape
    k = min(k, n)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k] if k < n else np.tile(np.arange(n), (m, 1))
    rows = np.arange(m)[:, None]
    idx = idx[rows, np.argsort(D[rows, idx], axis=1)]  # nearest first
    nb_labels = labels[idx]
    counts = (nb_labels[:, :, None] == np.arange(n_classes)).sum(axis=1)
    winners = counts == counts.max(axis=1, keepdims=True)
    tied = winners.sum(axis=1) > 1
    pred = counts.argmax(axis=1)
    if ties == "nearest":
        # among tied classes, pick the class of the closest neighbour
        first = winners[rows, nb_labels].argmax(axis=1)
        pred = nb_labels[np.arange(m), first]
    else:
        pred[tied] = -1
    return pred, idx


def decision_grid(X, labels, k, metric, n_classes, ties):
    g = np.linspace(-LIM, LIM, GRID_RES)
    gx, gy = np.meshgrid(g, g)
    Q = np.c_[gx.ravel(), gy.ravel()]
    pred = np.empty(len(Q), dtype=int)
    for s in range(0, len(Q), 4096):  # chunk to bound memory
        pred[s:s + 4096], _ = knn_vote(distances(Q[s:s + 4096], X, metric), labels, k, n_classes, ties)
    return g, pred.reshape(GRID_RES, GRID_RES)


def accuracies(X, labels, k, metric, n_classes, ties):
    """Training accuracy and leave-one-out accuracy (ties count as errors)."""
    D = distances(X, X, metric)
    train_pred, _ = knn_vote(D, labels, k, n_classes, ties)
    np.fill_diagonal(D, np.inf)
    loo_pred, _ = knn_vote(D, labels, k, n_classes, ties) if len(X) > 1 else (np.full(1, -1), None)
    return (train_pred == labels).mean(), (loo_pred == labels).mean()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(data, k, metric, n_classes, ties, query):
    X = np.c_[data["x"], data["y"]] if data["x"] else np.empty((0, 2))
    labels = np.asarray(data["label"], dtype=int)
    fig = go.Figure()

    # Decision regions: discrete colorscale over values -1 (tie) .. n_classes-1
    colors = [TIE_COLOR] + REGION_COLORS[:n_classes]
    n = len(colors)
    colorscale = []
    for i, c in enumerate(colors):
        colorscale += [[i / n, c], [(i + 1) / n, c]]
    if len(X):
        g, Z = decision_grid(X, labels, k, metric, n_classes, ties)
    else:
        g, Z = np.linspace(-LIM, LIM, GRID_RES), np.full((GRID_RES, GRID_RES), -1)
    fig.add_trace(go.Heatmap(
        x=g, y=g, z=Z, zmin=-1.5, zmax=n_classes - 0.5, colorscale=colorscale,
        showscale=False, hoverinfo="none", name="regions",
    ))

    # Training points, one trace per class
    for c in range(n_classes):
        m = labels == c
        fig.add_trace(go.Scatter(
            x=X[m, 0], y=X[m, 1], mode="markers", name=CLASS_NAMES[c],
            marker=dict(size=9, color=POINT_COLORS[c], line=dict(width=1, color="black")),
            hovertemplate=f"{CLASS_NAMES[c]}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
        ))

    # Query point with lines to its k nearest neighbours
    query_text = "Switch the click action to “Query” and click on the plot to inspect a location."
    if query is not None and len(X):
        q = np.asarray([query])
        pred, idx = knn_vote(distances(q, X, metric), labels, k, n_classes, ties)
        nb = idx[0]
        lx, ly = [], []
        for j in nb:
            lx += [q[0, 0], X[j, 0], None]
            ly += [q[0, 1], X[j, 1], None]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="black", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(
            x=X[nb, 0], y=X[nb, 1], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(size=17, color="rgba(0,0,0,0)", line=dict(width=2.5, color="black")),
        ))
        p = int(pred[0])
        fig.add_trace(go.Scatter(
            x=q[:, 0], y=q[:, 1], mode="markers", name="query", hoverinfo="skip",
            marker=dict(symbol="star", size=20, color=POINT_COLORS[p] if p >= 0 else "white",
                        line=dict(width=1.5, color="black")),
        ))
        counts = np.bincount(labels[nb], minlength=n_classes)
        votes = ", ".join(f"{CLASS_NAMES[c]}: {counts[c]}" for c in range(n_classes) if counts[c])
        verdict = CLASS_NAMES[p] if p >= 0 else "tie (undecided)"
        query_text = f"Query ({q[0, 0]:.2f}, {q[0, 1]:.2f}) — votes of {len(nb)} neighbours: {votes} → {verdict}"

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-LIM, LIM], constrain="domain", showgrid=False, zeroline=False),
        yaxis=dict(range=[-LIM, LIM], scaleanchor="x", showgrid=False, zeroline=False),
        plot_bgcolor="white", hovermode="closest", hoverdistance=6,
        legend=dict(orientation="h", y=-0.02, yanchor="top", x=0.5, xanchor="center"),
        uirevision="keep",  # preserve zoom/pan across updates
    )
    return fig, query_text


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def control(label, component):
    return html.Div([html.Label(label, style={"fontWeight": 600}), component], style={"marginBottom": "18px"})


app = Dash(__name__, title="k-NN demo")
app.layout = html.Div(
    style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "padding": "16px",
           "fontFamily": "system-ui, sans-serif", "maxWidth": "1200px", "margin": "auto"},
    children=[
        dcc.Store(id="data"),
        dcc.Store(id="seed", data=0),
        dcc.Store(id="query", data=None),
        html.Div(style={"flex": "1 1 300px", "maxWidth": "340px"}, children=[
            html.H2("k-Nearest Neighbours", style={"marginTop": 0}),
            control("Number of neighbours k", dcc.Slider(
                id="k", min=1, max=15, step=1, value=1,
                marks={i: str(i) for i in (1, 3, 5, 7, 9, 11, 13, 15)})),
            control("Distance metric", dcc.RadioItems(
                id="metric", value="L2", inline=True, inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                options=[{"label": "L1 (Manhattan)", "value": "L1"}, {"label": "L2 (Euclidean)", "value": "L2"}])),
            control("Ties in the vote", dcc.RadioItems(
                id="ties", value="undecided", inline=True,
                inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                options=[{"label": "show as white", "value": "undecided"},
                         {"label": "nearest neighbour wins", "value": "nearest"}])),
            html.Hr(),
            control("Dataset", dcc.Dropdown(
                id="dataset", value="blobs", clearable=False,
                options=[{"label": v, "value": k} for k, v in DATASETS.items()])),
            control("Number of classes", dcc.Slider(id="n-classes", min=2, max=5, step=1, value=3)),
            control("Points per class", dcc.Slider(
                id="n-points", min=5, max=100, step=5, value=20,
                marks={i: str(i) for i in (5, 25, 50, 75, 100)})),
            html.Div([
                html.Button("Randomize points", id="randomize", n_clicks=0),
                html.Button("Clear all points", id="clear", n_clicks=0, style={"marginLeft": "8px"}),
            ], style={"marginBottom": "18px"}),
            html.Hr(),
            control("Click on the plot to", dcc.RadioItems(
                id="click-mode", value="add",
                options=[{"label": " add a point", "value": "add"},
                         {"label": " remove the nearest point", "value": "remove"},
                         {"label": " query: show the k nearest neighbours", "value": "query"}])),
            control("Class of added points", dcc.RadioItems(id="add-class", value=0, inline=True)),
            html.Hr(),
            html.Div(id="stats"),
        ]),
        html.Div(style={"flex": "3 1 500px", "minWidth": "320px"}, children=[
            dcc.Graph(id="graph", style={"height": "75vh", "minHeight": "420px"},
                      config={"displaylogo": False}),
            html.Div(id="query-text", style={"marginTop": "6px", "color": "#444"}),
        ]),
    ],
)


@app.callback(
    Output("add-class", "options"),
    Output("add-class", "value"),
    Input("n-classes", "value"),
    State("add-class", "value"),
)
def update_class_options(n_classes, current):
    options = [{"label": html.Span(f" {CLASS_NAMES[c]}", style={"color": POINT_COLORS[c], "fontWeight": 600,
                                                                 "marginRight": "8px"}),
                "value": c} for c in range(n_classes)]
    return options, current if current is not None and current < n_classes else 0


@app.callback(
    Output("data", "data"),
    Output("seed", "data"),
    Output("query", "data"),
    Input("dataset", "value"),
    Input("n-classes", "value"),
    Input("n-points", "value"),
    Input("randomize", "n_clicks"),
    Input("clear", "n_clicks"),
    Input("graph", "clickData"),
    State("click-mode", "value"),
    State("add-class", "value"),
    State("data", "data"),
    State("seed", "data"),
    State("query", "data"),
    State("metric", "value"),
)
def update_data(dataset, n_classes, n_points, _rand, _clear, click, mode, add_class, data, seed, query, metric):
    trigger = ctx.triggered_id
    if trigger == "graph" and click and data is not None:
        pt = click["points"][0]
        x, y = float(pt["x"]), float(pt["y"])
        if mode == "add":
            data = {"x": data["x"] + [x], "y": data["y"] + [y], "label": data["label"] + [int(add_class)]}
        elif mode == "remove" and data["x"]:
            X = np.c_[data["x"], data["y"]]
            i = int(distances(np.array([[x, y]]), X, metric).argmin())
            data = {key: v[:i] + v[i + 1:] for key, v in data.items()}
        elif mode == "query":
            query = [x, y]
        return data, seed, query
    if trigger == "clear":
        return {"x": [], "y": [], "label": []}, seed, None
    if trigger == "randomize":
        seed += 1
    return make_dataset(dataset, n_classes, n_points, seed), seed, query


@app.callback(
    Output("graph", "figure"),
    Output("query-text", "children"),
    Output("stats", "children"),
    Input("data", "data"),
    Input("k", "value"),
    Input("metric", "value"),
    Input("ties", "value"),
    Input("query", "data"),
    State("n-classes", "value"),
)
def render(data, k, metric, ties, query, n_classes):
    if data is None:
        data = {"x": [], "y": [], "label": []}
    # a point added with a class beyond the current number of classes is impossible,
    # but data generated before a class-count change might be; guard anyway
    n_classes = max([n_classes] + [l + 1 for l in data["label"]])
    fig, query_text = build_figure(data, k, metric, n_classes, ties, query)
    n = len(data["x"])
    if n:
        X = np.c_[data["x"], data["y"]]
        train_acc, loo_acc = accuracies(X, np.asarray(data["label"]), k, metric, n_classes, ties)
        stats = [html.Div(f"{n} training points"),
                 html.Div(f"Training accuracy: {train_acc:.1%}"),
                 html.Div(f"Leave-one-out accuracy: {loo_acc:.1%}")]
    else:
        stats = [html.Div("No training points — click on the plot to add some.")]
    stats.append(html.P(
        "White regions are where the vote among the k neighbours is tied. "
        "With k = 1 the training accuracy is (almost) always 100%; the leave-one-out "
        "accuracy is a better estimate of generalisation.",
        style={"color": "#666", "fontSize": "0.9em"}))
    return fig, query_text, stats


if __name__ == "__main__":
    app.run(debug=True)
