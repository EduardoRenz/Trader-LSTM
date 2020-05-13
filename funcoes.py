
from datetime import timedelta
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU,BatchNormalization,Bidirectional

from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


CORRETORAS = pd.read_csv('./CorBov.txt',header=None,names=['codigo','nome'],sep=';') # pegar DF das corretoras

# Cor dos agressores
AGRESSORS_COLOR= {'vendedor':'red','direto':'grey','comprador':'green'}


#Creates a multi-dimension dataset for LSTM networks
def create_lstm_dataset(X, y, time_steps=50):
  Xs, ys = [], []
  for i in range(len(X) - time_steps):
      v = X.iloc[i:(i + time_steps)].values
      Xs.append(v)
      ys.append(y.iloc[i + time_steps])
  return np.array(Xs), np.array(ys)


#Cria o grafico principal
def buildMainGraph(negocios):

  #volume e VAP
  volume = negocios.groupby('agressor')['qtd'].resample('5Min').sum()
  vap = negocios.groupby(['agressor','preco'])['qtd'].sum()
  fig = make_subplots(rows=2,cols=2, shared_yaxes=True,shared_xaxes=True,column_widths=[0.2, 0.8],row_heights=[0.8, 0.2], horizontal_spacing = 0,vertical_spacing=0)
  #Add a linha do preço
  fig.add_trace(go.Scatter( x=negocios.index, y=negocios.preco, name="Preço"  ),row=1,col=2)

  #Volume separado por seus agressores
  for (agressor,v) in volume.groupby('agressor'):
    fig.add_trace(go.Bar( x=v[agressor].index, y=v[agressor].values, name=f"Volume {agressor}",marker={'color':AGRESSORS_COLOR[agressor]}),row=2,col=2)

  #VAP separado por seus agressores
  for (agressor,v) in vap.groupby('agressor'):
    fig.add_trace(go.Bar( x=vap[agressor], y=vap[agressor].index,orientation='h',name=f"VAP {agressor}",marker={'color':AGRESSORS_COLOR[agressor]}),row=1,col=1)

  fig.update_layout(barmode='stack',margin=dict(r=10, t=10, b=10, l=10),width=1200)

  return fig

#Carrega, trata e gera pd dos negocios
def loadNegocios(path):
  negocios = pd.read_csv(path,sep=';',header=None,names=['preco','qtd','datetime','comprador','vendedor','cod_agressor']) # Carregar os negocios
  #Renomear algumas colunas
  negocios.rename(columns={"vendedor":"cod_vendedor"},inplace=True)
  negocios.rename(columns={"comprador":"cod_comprador"},inplace=True)
  #merge para pegar o nome das corretoras
  negocios = negocios.merge(CORRETORAS,left_on="cod_vendedor",right_on="codigo",how='left').rename(columns={"nome":"vendedor"}) # merges com corretora
  negocios = negocios.merge(CORRETORAS,left_on="cod_comprador",right_on="codigo",how='left').rename(columns={"nome":"comprador"}) # merges com corretora
  #Definir que corretora é uma categoria
  negocios.vendedor = negocios.vendedor.astype('category') # transformar corretora em categoria
  negocios.comprador = negocios.comprador.astype('category') # transformar corretora em cateogria
  #Atribuir inidce sendo o tempo
  negocios.index =pd.to_datetime(negocios.datetime , format='%Y%m%d%H%M%S') # definir indice como sendo o datetime
  # Desduplicar segundos no times and trades adicionando ms
  negs_p_segundo = negocios.groupby(level=0).cumcount()
  negocios.index = negocios.index + pd.to_timedelta(negs_p_segundo, unit='ms')
  #Preenche lacunas de tempo com milisegundos
  full_day = pd.date_range(negocios.index.min(),negocios.index.max(),freq='ms')
  negocios_full_day = negocios.reindex(full_day,fill_value=None)
  #negocios.resample('10ms').fillna(None)
  # Testar : Definindo quem foi agressor
  negocios.loc[negocios.cod_agressor == 1,'agressor'] = 'comprador'
  negocios.loc[negocios.cod_agressor == 2,'agressor'] = 'vendedor'
  negocios.loc[negocios.cod_agressor == 4,'agressor'] = 'direto'
  #Drop de colunas desnecessarias
  negocios = negocios.drop('datetime',axis=1)
  negocios = negocios.drop(['codigo_x','codigo_y','cod_vendedor','cod_comprador','cod_agressor'],axis=1) # drop de colunas desnecessarias


  #Data engeenering
  negocios['vap'] = negocios.groupby(['preco'])['qtd'].transform('sum')
  negocios['vap_5min'] = negocios.rolling('5min',min_periods=1).qtd.sum()
  negocios['vap_comprador'] = negocios.query("agressor == 'comprador'").groupby(['preco'])['qtd'].transform('sum')
  negocios['vap_vendedor'] = negocios.query("agressor == 'vendedor'").groupby(['preco'])['qtd'].transform('sum')
  negocios['vap_direto'] = negocios.query("agressor == 'direto'").groupby(['preco'])['qtd'].transform('sum')

  negocios.vap_vendedor.fillna(method='pad',inplace=True)
  negocios.vap_comprador.fillna(method='pad',inplace=True)
  negocios.vap_direto.fillna(method='pad',inplace=True)

  negocios['vap_5min_comprador'] = negocios.query("agressor == 'comprador'").rolling('5min',min_periods=1)['qtd'].sum()
  negocios['vap_5min_vendedor'] = negocios.query("agressor == 'vendedor'").rolling('5min',min_periods=1)['qtd'].sum() 
  negocios['vap_5min_direto'] = negocios.query("agressor == 'direto'").rolling('5min',min_periods=1)['qtd'].sum() 


  negocios.vap_5min_vendedor.fillna(method='pad',inplace=True)
  negocios.vap_5min_comprador.fillna(method='pad',inplace=True)
  negocios.vap_5min_direto.fillna(method='pad',inplace=True)

  negocios.fillna(0,inplace=True)

  #Aqui o que o trader deve fazer
  negocios['acao'] = 'do_nothing'
  negocios['acao'] = negocios['acao'].astype('category')
  negocios.acao.cat.set_categories(['do nothing','buy','sell'],inplace=True)
  return negocios

#Gera dummies já com todas as corretoras
def negociosWithDummies(negocios):
  X_columns = pd.get_dummies(negocios.drop(columns=['acao'])).columns # retirado 'ano' 
  y_columns = pd.get_dummies(negocios[['acao']]).columns

  df = negocios.copy()

  for comprador in list(filter(lambda x : x not in (negocios.comprador.values)  , CORRETORAS.values[:,1])):
    df['comprador_'+comprador] = 0

  for vendedor in list(filter(lambda x : x not in (negocios.vendedor.values)  , CORRETORAS.values[:,1])):
    df['vendedor_'+vendedor] = 0

  #One hot dos dados
  df = pd.get_dummies(df)
  df = df.sort_index(axis=1)

  return df

#Sugere entradas de compra e venda na coluna acao
def sugestEntrances(negocios):
  #pontos de entrada automatico
  negocios['acao'] = 'do_nothing' # reset

  current_moment = negocios.index.min()
  last_moment = current_moment
  delta = timedelta(minutes=4) # steps que vao avançar 
  look_forward = timedelta(minutes=5) #Quanto ira olhar para frente para ver qual será o preço
  spread_minimo = 2 # minimo de pontos que tem que variar para entarr na operacao

  #avança i vezes para frente verificando os preços
  for i in range(180):
    forward = negocios.loc[(negocios.index >= current_moment) & (negocios.index <= current_moment +look_forward) ]
    minimo = forward.preco.min()
    maximo = forward.preco.max()
    forward_min = forward[forward.preco == minimo ].index
    forward_max = forward[forward.preco == maximo ].index

    if len(forward)> 0:
      forward = forward[forward.preco == minimo ].index[0]
    else:
      continue

    entrada_compra = forward_min[0]
    entrada_venda = forward_max[0]
    spread = maximo - minimo

    if(spread >= spread_minimo):
      negocios.loc[entrada_compra == negocios.index,'acao'  ] = 'buy'
      negocios.loc[entrada_venda == negocios.index,'acao'  ] = 'sell'
  

    last_moment = current_moment
    current_moment += delta


  return negocios


# Oversample para equilibrar as classes
def oversample(X,y):
  sm = SMOTE(random_state=42)
  X_res, y_res = sm.fit_resample(X.values,y.values)
  X_res_df = pd.DataFrame(X_res,columns=X.columns)
  y_res_df = pd.DataFrame(y_res,columns=y.columns)
  return X_res_df,y_res_df





