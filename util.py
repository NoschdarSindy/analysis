import glob

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
from matplotlib import pyplot as plt
import neurokit2 as nk
import seaborn as sns


def g(subject, glob_pattern="*"):
    return glob.glob(f"data/{subject}/{glob_pattern}")


# Helper function to get a color from the Plotly color palette
def get_color(i):
    colors = px.colors.qualitative.Plotly
    return colors[i % len(colors)]


def markers_to_gantt(markers_df):
    marker_prefixes = markers_df.marker.str.split("/").str[0:2].apply("/".join).unique()

    def get_index(marker_prefix, marker_suffix):
        matches = markers_df.loc[
            markers_df.marker.str.contains(f"{marker_prefix}/{marker_suffix}")
        ]
        if matches.empty:
            if marker_suffix == "start":
                return markers_df.index[0]
            else:
                return markers_df.index[-1]

        return matches.index[0]

    is_datetime = isinstance(markers_df.index, pd.DatetimeIndex)
    result = []
    for marker_prefix in marker_prefixes:
        start = get_index(marker_prefix, "start")
        end = get_index(marker_prefix, "end")
        duration = end - start
        if is_datetime:
            duration = duration.total_seconds()
        result.append(
            dict(
                marker=marker_prefix,
                start=start,
                end=end,
                duration=duration,
            )
        )
    return result


# Helper function to plot streams and markers using plotly
def plot_channels(streams_df, markers_df, title=None, hide_end_markers=False):
    upper_fig = go.Figure()

    # Streams
    for i, col in enumerate(streams_df):
        trace = go.Scatter(
            x=streams_df.index,
            y=streams_df[col],
            mode="lines",
            name=col,
            yaxis=f"y{i + 1}",
            line=dict(color=get_color(i)),
        )
        upper_fig.add_trace(trace)

    # Markers
    filtered_markers_df = (
        markers_df.loc[markers_df.marker.str.contains("/start")]
        if hide_end_markers
        else markers_df
    )
    markers_x = filtered_markers_df.index
    upper_fig.add_trace(
        go.Scatter(
            x=markers_x,
            y=[0] * len(filtered_markers_df),
            yaxis=f"y{len(streams_df.columns) + 1}",
            mode="markers",
            marker=dict(
                symbol="diamond-tall",
                line=dict(color="black", width=2),
                color="yellow",
                size=14,
            ),
            customdata=filtered_markers_df[[*filtered_markers_df][1:-1]],
            showlegend=True,
            name="Start Markers" if hide_end_markers else "Markers",
            hovertext=[
                " : ".join(map(str, i))
                for i in zip(
                    (
                        filtered_markers_df.index.strftime("%M:%S:%f")
                        if isinstance(filtered_markers_df.index, pd.DatetimeIndex)
                        else filtered_markers_df.index
                    ),
                    filtered_markers_df.marker,
                )
            ],
            hoverinfo="text",
        )
    )

    layout = go.Layout(
        margin=dict(b=0, pad=0),
        title=title,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", yanchor="bottom"),
        xaxis=dict(
            title=dict(text="Time", standoff=0),
            tickformat="%M:%S:%f",
            range=[markers_df.index.min(), markers_df.index.max()],
            side="top",
        ),
        hovermode="x unified",
        **{
            f"yaxis{len(streams_df.columns) + 1}": dict(
                title=dict(standoff=0), overlaying="y", visible=False, range=[-0.05, 1]
            )
        },
        **{
            f"yaxis{i + 1}": dict(
                title=dict(text=col, standoff=0),
                overlaying="y" if i > 0 else None,
                visible=False,
                color=get_color(i),
            )
            for i, col in enumerate(streams_df)
        },
    )
    upper_fig.update_layout(layout)
    upper_fig.show()


def plot_gantt(markers_df):
    # Markers as Gantt chart
    gantt_data = markers_to_gantt(markers_df)
    gantt_fig = px.timeline(
        gantt_data,
        x_start="start",
        x_end="end",
        y="marker",
        color="marker",
        hover_data={"duration": ":.2f"},
    )
    gantt_fig.update_layout(
        legend=dict(y=-0.1, x=0.5, orientation="h", xanchor="center"),
        margin=dict(t=0, pad=0),
        xaxis=dict(
            title=dict(text="Time", standoff=0),
            tickformat="%M:%S:%f",
            range=[markers_df.index.min(), markers_df.index.max()],
        ),
        yaxis=dict(showticklabels=False, title="Markers"),
    )
    gantt_fig.show()


def plot_data(streams_df, markers_df, title=None, hide_end_markers=False):
    plot_channels(
        streams_df, markers_df, title=title, hide_end_markers=hide_end_markers
    )
    plot_gantt(markers_df)


def plot_epoch(
    epoch,
    title=None,
    columns_to_plot=[
        # "ECG_Rate",
        "EDA_Tonic",
        "EDA_Phasic",
        # "pupil_diameter"
    ],
    subplots=False,
):
    nk.signal_plot(
        epoch[columns_to_plot],
        # can only pass title arg if not subplots
        **dict(title=title or epoch["Condition"].values[0]) if not subplots else {},
        # Extract condition name
        subplots=subplots,
        labels=columns_to_plot,
    )


def plot_correlation_matrix(correlation_matrix, title=None):
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    correlation_matrix = correlation_matrix.mask(mask)
    plt.figure(figsize=(0.55 * len(correlation_matrix), 0.25 * len(correlation_matrix)))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()


samples_to_seconds = lambda samples, sampling_rate: samples / sampling_rate
seconds_to_samples = lambda seconds, sampling_rate: int(seconds * sampling_rate)
