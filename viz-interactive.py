import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc
#import dash_core_components as dcc
#import dash_html_components as html
from dash.dependencies import Input, Output

# developper avec chatgpt, ajouter seconde variable pour scatterplot

# Step 1: Load the data from the CSV file
data = pd.read_csv('SeoulBikeData.csv')

# Step 2: Create the Dash app
app = dash.Dash(__name__)

# Step 3: Create the layout for the app
app.layout = html.Div([
    html.H1('Variable Selection Dashboard'),
    dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': col, 'value': col} for col in data.columns],
        value=data.columns[0]
    ),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='histogram')
])

# Step 4: Define the callback functions
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_scatter_plot(selected_variable):
    fig = px.scatter(data, x=selected_variable, y='y_column')
    fig.update_layout(
        xaxis=dict(title=selected_variable),
        yaxis=dict(title='Y-axis')
    )
    return fig

@app.callback(
    Output('histogram', 'figure'),
    [Input('variable-dropdown', 'value')]
)
def update_histogram(selected_variable):
    fig = px.histogram(data, x=selected_variable)
    fig.update_layout(
        xaxis=dict(title=selected_variable),
        yaxis=dict(title='Frequency')
    )
    return fig

# Step 5: Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
    
