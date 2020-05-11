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

pd.set_option('precision', 6)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# %%
trader = Trader('trade_oversampled.h5')
#%%
trader.loadTrainData('./dolar/negocios/20200220_dolh20')

#%% Treinar ou Re-Treinar o modelo
trader.model.load_weights('./dolar/dolar_oversampled.h5')
trader.fit()

#%%
trader.loadTestData('./dolar/negocios/20200427_dolk20')

#%%
trader.buildModel(trader.test_shape)
trader.model.build(trader.test_shape)
trader.model.load_weights('./dolar/dolar_oversampled.h5')
#%%
predictions = trader.predict(trader.test_data[0])
# %%
trader.model.evaluate(trader.test_data[0],predictions)
trader.plotPredictions(predictions)
# %%
