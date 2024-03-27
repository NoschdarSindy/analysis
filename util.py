import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns


# Helper function to get a color from the Plotly color palette
def get_color(i):
    colors = px.colors.qualitative.Plotly
    return colors[i % len(colors)]


def markers_to_gantt(markers_df):
    marker_prefixes = markers_df.marker.str.split('-').str[0].unique()

    def get_index(marker_prefix, marker_suffix):
        matches = markers_df.loc[markers_df.marker.eq(f'{marker_prefix}-{marker_suffix}')]
        if matches.empty:
            if marker_suffix == 'start':
                return markers_df.index[0]
            else:
                return markers_df.index[-1]
        return matches.index[0]

    return [dict(marker=marker_prefix, start=get_index(marker_prefix, 'start'),
                 end=get_index(marker_prefix, 'end')) for marker_prefix in marker_prefixes]

# Helper function to plot streams and markers using plotly
def plot_channels(streams_df, markers_df, title=None, hide_end_markers=False):
    upper_fig = go.Figure()

    # Streams
    for i, col in enumerate(streams_df):
        trace = go.Scatter(x=streams_df.index, y=streams_df[col], mode='lines', name=col, yaxis=f'y{i + 1}',
                           line=dict(color=get_color(i)))
        upper_fig.add_trace(trace)

    # Markers
    filtered_markers_df = markers_df.loc[markers_df.marker.str.endswith('-start')] if hide_end_markers else markers_df
    markers_x = filtered_markers_df.index
    upper_fig.add_trace(go.Scatter(x=markers_x, y=[0] * len(filtered_markers_df), yaxis=f'y{len(streams_df.columns) + 1}',
                                   mode='markers',
                                   marker=dict(symbol='diamond-tall', line=dict(color='black', width=2), color='yellow',
                                               size=14),
                                   customdata=filtered_markers_df[[*filtered_markers_df][1:-1]],
                                   showlegend=True,
                                   name='Start Markers' if hide_end_markers else 'Markers',
                                   hovertext=[' : '.join(map(str, i)) for i in
                                              zip(filtered_markers_df.index.strftime("%M:%S:%f") if isinstance(filtered_markers_df.index, pd.DatetimeIndex) else filtered_markers_df.index, filtered_markers_df.marker)],
                                   hoverinfo='text',
                                   ))

    layout = go.Layout(
        margin=dict(b=0, pad=0),
        title=title,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center', yanchor='bottom'),
        xaxis=dict(
            title=dict(
                text='Time',
                standoff=0),
            tickformat="%M:%S:%f", range=[markers_df.index.min(), markers_df.index.max()],
            side='top'),
        hovermode="x unified",
        **{f'yaxis{len(streams_df.columns) + 1}': dict(title=dict(standoff=0), overlaying='y', visible=False, range=[-0.05, 1])},

        **{f'yaxis{i + 1}': dict(title=dict(text=col, standoff=0), overlaying='y' if i > 0 else None,
                                 visible=False,
                                 color=get_color(i)) for i, col in enumerate(streams_df)},
    )
    upper_fig.update_layout(layout)
    pyo.iplot(upper_fig)


def plot_gantt(markers_df):
    # Markers as Gantt chart
    gantt_data = markers_to_gantt(markers_df)
    gantt_fig = px.timeline(gantt_data, x_start='start', x_end='end', y='marker', color='marker')
    gantt_fig.update_layout(
        legend=dict(y=-0.1, x=0.5, orientation="h", xanchor='center'),
        margin=dict(t=0, pad=0), xaxis=dict(title=dict(
            text='Time',
            standoff=0), tickformat="%M:%S:%f", range=[markers_df.index.min(), markers_df.index.max()]),
        yaxis=dict(showticklabels=False, title='Markers'))

    pyo.iplot(gantt_fig)


def plot_data(streams_df, markers_df, title=None, hide_end_markers=False):
    plot_channels(streams_df, markers_df, title=title, hide_end_markers=hide_end_markers)
    plot_gantt(markers_df)


def plot_correlation_matrix(correlation_matrix, title=None):
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    correlation_matrix = correlation_matrix.mask(mask)
    plt.figure(figsize=(0.55 * len(correlation_matrix), 0.25 * len(correlation_matrix)))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()
    # fig = px.imshow(correlation_matrix, color_continuous_scale='RdBu', text_auto='.2f')
    # fig.update_layout(title=title, title_x=0.5, height=1000)
    # fig.show()