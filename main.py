#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import datetime as dt

from funcoes import *
from Trader import Trader


# %%
trader = Trader('trade_oversampled.h5')
trader.loadTrainData('./dolar/negocios/20200220_dolh20')

#%%
trader.fit()

#%%
trader.loadTestData('./dolar/negocios/20200220_dolh20')

#%%
trader.model.load_weights('./dolar/dolar_oversampled.h5')
#%%
predictions = trader.model.predict(trader.test_data[0])
# %%
trader.model.evaluate(trader.test_data[0],predictions)

# %%
