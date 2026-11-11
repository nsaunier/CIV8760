# k-NN and k-means interactive demos

Two interactive teaching demos written in Python with [Dash](https://dash.plotly.com/) and
[Plotly](https://plotly.com/python/):

- **`knn_demo.py`** — k-nearest neighbours classification, inspired by the
  [Stanford CS231n k-NN demo](https://web.archive.org/web/20251204022137/http://vision.stanford.edu/teaching/cs231n-demos/knn/).
  Labelled 2D points are shown on top of the decision regions of a k-NN classifier.
- **`kmeans_demo.py`** — k-means clustering (Lloyd's algorithm), run one step at a time,
  alternating between assigning points to the nearest centroid and moving the centroids.

## Installation

Python 3.10 or later is needed. Install the dependencies (dash, plotly, numpy) in a virtual
environment, for example with [uv](https://docs.astral.sh/uv/):

```
uv venv .venv
uv pip install -p .venv -r requirements.txt
```

or with the standard tools:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

(On Windows, use `.venv\Scripts\python` and `.venv\Scripts\pip` instead of `.venv/bin/...`.)

## Running the demos

Each demo is a standalone script that starts a local web server:

```
.venv/bin/python knn_demo.py
.venv/bin/python kmeans_demo.py
```

Then open <http://127.0.0.1:8050> in a browser. Stop the server with Ctrl+C. Both demos use
port 8050 by default, so run one at a time, or change the port in the last line of the script,
e.g. `app.run(debug=True, port=8051)`.

### In a Jupyter notebook

Both demos can also be shown inside a notebook (JupyterLab, Jupyter Notebook, VS Code or
Google Colab). From a notebook in the same folder as the scripts, with dash, plotly and numpy
installed in the notebook's kernel:

```python
from knn_demo import app          # or: from kmeans_demo import app
app.run(jupyter_mode="inline")    # embed the demo in the cell output
```

Other modes are `"external"` (open in a browser tab), `"tab"` and `"jupyterlab"`. In Colab,
first run `!pip install dash` and upload the script. The demo needs a running kernel: it is not
interactive in a static export of the notebook.

## Using the k-NN demo

- **k** (1–15) and the **distance metric** (L1 or L2) update the decision regions immediately.
- **Ties** in the vote are either shown as white regions (as in the CS231n demo) or broken in
  favour of the class of the nearest neighbour.
- **Dataset**: Gaussian blobs, uniform random points with random labels, concentric rings or
  spirals, with 2–5 classes and 5–100 points per class. "Randomize points" draws a new sample,
  "Clear all points" removes everything.
- **Clicking on the plot** either adds a point of the selected class, removes the nearest
  point, or queries a location: the query shows its k nearest neighbours and their vote.
- The side panel shows the training accuracy and the leave-one-out accuracy.

## Using the k-means demo

- **k** (1–8) and the **initialisation** (random data points, uniformly random positions, or
  k-means++) restart the algorithm from new centroids.
- **Step** runs the next half-iteration; its label shows whether the next step assigns the
  points or updates the centroids. **Run to convergence** runs all remaining steps,
  **Play/Pause** runs them automatically, and **Re-initialise centroids** starts over from new
  random centroids. The algorithm has converged when an assignment step changes nothing.
- The background shows the region closest to each centroid and the lines trace the centroid
  paths; both can be turned off.
- **Dataset**: Gaussian blobs, elongated blobs, blobs of uneven size, concentric rings or
  uniform random, with a chosen number of underlying groups and points.
- **Clicking on the plot** adds a point, which stays grey until the next assignment step.
- The side panel shows the number of iterations, the inertia (sum of squared distances of the
  points to their centroid) and whether the algorithm has converged.
