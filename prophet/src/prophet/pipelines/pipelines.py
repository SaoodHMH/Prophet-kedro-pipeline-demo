from kedro.pipeline import node, Pipeline
from ..prophet.nodes.functions import *

def simple_prophet(**kwargs):
    return Pipeline(
        [
            node(
                func=run_prophet,
                inputs="Manning_df",
                outputs="forecast_model"
            ),
            node(
                func=forecast,
                inputs="forecast_model",
                outputs="forecast_df"
            ),
            node(
                func=plot_forecast,
                inputs= {"model":"forecast_model", "forecast":"forecast_df"},
                outputs="forecast_plot"
            ),
            node(
                func=plot_components,
                inputs= {"model":"forecast_model", "forecast":"forecast_df"},
                outputs="component_plot"
            )
        ]
    )

register_pipeline("simple_prophet", simple_prophet)