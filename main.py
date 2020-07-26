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
from model import *


pd.set_option('precision', 6)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# %%
trader = Trader()
#%%
trader.loadTrainData('./dolar/negocios/20200220_dolh20')
print(trader.X_train_df_scaled.tail())

#%% Treinar ou Re-Treinar o modelo
#trader.model.load_weights('./dolar/dolar_oversampled.h5')
trader.buildModel(trader.train_shape)
trader.fit()

# #%%
#trader.loadTestData('./dolar/negocios/20200427_dolk20')
#print(trader.X_test_df_scaled.head())
# #%%
# trader.buildModel(trader.test_shape)
# trader.model.build(trader.test_shape)
# trader.model.load_weights('./dolar/dolar_oversampled.h5')
# #%%
# predictions = trader.predict(trader.test_data[0])
# # %%
# trader.model.evaluate(trader.test_data[0],predictions)
# trader.plotPredictions(predictions)
# %%
