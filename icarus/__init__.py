"""
icarus
=========
Data-driven heat flux prediction from infrared thermography.

Provides a full pipeline from raw IR data to trained heat flux prediction
models using Proper Orthogonal Decomposition (POD), Dynamic Mode
Decomposition (DMD), and artificial neural networks.

Quickstart
----------
>>> import icarus as tf
>>> import numpy as np
>>>
>>> # Load your IR temperature and heat flux arrays [ny, nx, nt]
>>> T = np.load("temperature.npy")
>>> q = np.load("heatflux.npy")
>>>
>>> # Run the full pipeline
>>> pipeline = tf.Pipeline()
>>> pipeline.fit(T, q)
>>> q_pred = pipeline.predict(T_new)

Modules
-------
data            : Loading and preprocessing of IR thermography datasets
decomposition   : POD and DMD reduced-order analysis
features        : Feature construction (gradients, modal contributions)
models          : ANN training with Bayesian hyperparameter optimisation
metrics         : R², RMSE, MAE evaluation utilities
visualisation   : Spatial field, mode, and diagnostic plots
pipeline        : End-to-end Pipeline orchestrator
"""

from icarus.pipeline.runner import Pipeline
from icarus.pipeline.bandwise import BandwiseModalModel
from icarus.decomposition.pod import POD
from icarus.decomposition.dmd import DMD
from icarus.decomposition.spod import SPOD
from icarus.data.preprocessor import Preprocessor
from icarus.data.loader import load, load_field
from icarus.features.partition import partition_by_frequency, Partition
from icarus.models.neural import HeatFluxNet, SEARCH_SPACES
from icarus.models.probabilistic import ProbabilisticHeatFluxNet
from icarus.metrics.evaluation import evaluate, interval_metrics

__version__ = "0.5.0"
__author__ = "Raymond Twum-Barima"

__all__ = [
    "Pipeline",
    "BandwiseModalModel",
    "POD",
    "DMD",
    "SPOD",
    "Preprocessor",
    "load",
    "load_field",
    "partition_by_frequency",
    "Partition",
    "HeatFluxNet",
    "ProbabilisticHeatFluxNet",
    "SEARCH_SPACES",
    "evaluate",
    "interval_metrics",
    "__version__",
]
