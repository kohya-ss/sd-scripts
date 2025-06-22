import argparse
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    parser = argparse.ArgumentParser(description="Create dashboard from training log")
    parser.add_argument("--logfile", required=True, help="path to log csv file")
    parser.add_argument("--ma", type=int, default=100, help="window size for moving average of loss")
    args = parser.parse_args()

    df = pd.read_csv(args.logfile)

    # clean columns
    for col in ["Gradient Norm", "Threshold", "Loss", "ThreshOff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Scale" in df.columns:
        df["Scale"] = pd.to_numeric(df["Scale"], errors="coerce")
    if "CosineSim" in df.columns:
        df["CosineSim"] = pd.to_numeric(df["CosineSim"], errors="coerce")

    # replace NaN with 0 for specific columns
    for col in ["Gradient Norm", "Threshold", "CosineSim"]:
        if col in df.columns:
            df[col] = df[col].replace(np.nan, 0)

    # compute moving average for loss
    loss_ma = df["Loss"].rolling(window=args.ma, min_periods=1).mean()

    # create subplots
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        subplot_titles=["Gradient Norm and Threshold",
                                        "Loss (Moving Average)",
                                        "ThreshOff",
                                        "Scale",
                                        "CosineSim"])

    steps = df.index
    if "Gradient Norm" in df.columns:
        fig.add_trace(go.Scatter(x=steps, y=df["Gradient Norm"],
                                 name="Gradient Norm"), row=1, col=1)
    if "Threshold" in df.columns:
        fig.add_trace(go.Scatter(x=steps, y=df["Threshold"],
                                 name="Threshold"), row=1, col=1)

    fig.add_trace(go.Scatter(x=steps, y=loss_ma, name="Loss MA"), row=2, col=1)

    if "ThreshOff" in df.columns:
        fig.add_trace(go.Scatter(x=steps, y=df["ThreshOff"], name="ThreshOff"), row=3, col=1)

    if "Scale" in df.columns:
        fig.add_trace(go.Scatter(x=steps, y=df["Scale"], name="Scale"), row=4, col=1)
    else:
        fig.add_annotation(text="Scale column not found", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, row=4, col=1)

    if "CosineSim" in df.columns:
        fig.add_trace(go.Scatter(x=steps, y=df["CosineSim"], name="CosineSim"), row=5, col=1)
    else:
        fig.add_annotation(text="CosineSim column not found", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, row=5, col=1)

    fig.update_layout(height=1500, width=1000, showlegend=True, title=os.path.basename(args.logfile))

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.logfile))[0]
    out_file = os.path.join(out_dir, f"{base}_dashboard.html")
    fig.write_html(out_file, include_plotlyjs=True, full_html=True)
    print(f"Dashboard saved to {out_file}")


if __name__ == "__main__":
    main()
