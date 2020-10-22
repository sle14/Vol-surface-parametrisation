from numpy.lib.recfunctions import unstructured_to_structured as struct
from numpy.lib.recfunctions import structured_to_unstructured as unstruct
from numpy import log,sqrt,exp,inf,nan,pi
import plotly.graph_objects as go
from ipywidgets import widgets
import traitlets as tl
import pandas as pd
import numpy as np
import calibrator as cb
import grapher as gr
import pricer
import query
import utils

cout = utils.log(__file__,__name__,disp=False)
base = 253

symbol = "CGC"
qdate = "25/09/2020"
qtime = 2040

# ld = gr.LoadData(symbol,qdate,qtime)

# cs = ld.start()

# rs = cs.rs


params = cb.surface(symbol,qdate,qtime,errors=False,post=True)