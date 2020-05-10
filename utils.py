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
    fig.add_trace(go.Bar( x=v[agressor].index, y=v[agressor].values, name=f"Volume {agressor}",marker={'color':agressor_color[agressor]}),row=2,col=2)

  #VAP separado por seus agressores
  for (agressor,v) in vap.groupby('agressor'):
    fig.add_trace(go.Bar( x=vap[agressor], y=vap[agressor].index,orientation='h',name=f"VAP {agressor}",marker={'color':agressor_color[agressor]}),row=1,col=1)

  fig.update_layout(barmode='stack',margin=dict(r=10, t=10, b=10, l=10),width=1200)

  return fig

#Carrega, trata e gera pd dos negocios
def loadNegocios(path):
  negocios = pd.read_csv(path,sep=';',header=None,names=['preco','qtd','datetime','comprador','vendedor','cod_agressor']) # Carregar os negocios
  #Renomear algumas colunas
  negocios.rename(columns={"vendedor":"cod_vendedor"},inplace=True)
  negocios.rename(columns={"comprador":"cod_comprador"},inplace=True)
  #merge para pegar o nome das corretoras
  negocios = negocios.merge(corretoras,left_on="cod_vendedor",right_on="codigo",how='left').rename(columns={"nome":"vendedor"}) # merges com corretora
  negocios = negocios.merge(corretoras,left_on="cod_comprador",right_on="codigo",how='left').rename(columns={"nome":"comprador"}) # merges com corretora
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

  #Aqui o que o trader deve fazer
  negocios['acao'] = 'do nothing'
  negocios['acao'] = negocios['acao'].astype('category')
  negocios.acao.cat.set_categories(['do nothing','buy','sell'],inplace=True)
  return negocios

#Gera dummies já com todas as corretoras
def negociosWithDummies(negocios):
  X_columns = pd.get_dummies(negocios.drop(columns=['acao'])).columns # retirado 'ano' 
  y_columns = pd.get_dummies(negocios[['acao']]).columns

  df = negocios.copy()

  for comprador in list(filter(lambda x : x not in (negocios.comprador.values)  , corretoras.values[:,1])):
    df['comprador_'+comprador] = 0

  for vendedor in list(filter(lambda x : x not in (negocios.vendedor.values)  , corretoras.values[:,1])):
    df['vendedor_'+vendedor] = 0

  #One hot dos dados
  df = pd.get_dummies(df)
  df = df.sort_index(axis=1)

  return df

#Sugere entradas de compra e venda na coluna acao
def sugestEntrances(negocios):
  #pontos de entrada automatico
  negocios['acao'] = 'do nothing' # reset

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


# Prepara os treinos e testes
def createTrainTestDays(trainDay,testDay):
  trainDay =  sugestEntrances(trainDay)
  trainDay = negociosWithDummies(trainDay)
  testDay = negociosWithDummies(testDay)
  return trainDay,testDay


def splitXy(negocios,test_percent=80):
  y_columns = pd.get_dummies(negocios[['acao']]).columns
  X_columns =list(filter(lambda x : x not in y_columns,negocios.columns))
  #Train e test
  train_data = negocios.iloc[: int((len(negocios) /100) * test_percent)]
  X = train_data[X_columns]
  y= train_data[y_columns]

  return X,y

#Cria para lstm
def createTrainTestLSTM(X,y,time_steps=50):
  #Gera dados de treino e teste
  X_train_lstm, y_train_lstm = create_lstm_dataset(X,y, time_steps)
  X_test_lstm, y_test_lstm = create_lstm_dataset(X,y, time_steps)

  print(X_train_lstm.shape, y_train_lstm.shape)
  print(X_test_lstm.shape, y_test.shape)

  return X_train_lstm,y_train_lstm
    

# Oversample para equilibrar as classes
def oversample(X,y):
  sm = SMOTE(random_state=42)
  X_res, y_res = sm.fit_resample(X.values,y.values)
  X_res_df = pd.DataFrame(X_res,columns=X.columns)
  y_res_df = pd.DataFrame(y_res,columns=y.columns)
  return X_res_df,y_res_df



def buildModel(input_shape):
  dropout = 0.2 # dropout para cada lstm


  model = keras.Sequential()
  #model.add(keras.layers.LSTM(8, input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)))
  model.add(LeakyReLU())
  model.add(Dropout(dropout))
  model.add(BatchNormalization())

  model.add(Bidirectional(LSTM(128,return_sequences=True)))
  model.add(LeakyReLU())
  model.add(Dropout(dropout))
  model.add(BatchNormalization())

  #model.add(LSTM(128,return_sequences=True))
  #model.add(LeakyReLU())
  #model.add(Dropout(dropout))
  #model.add(BatchNormalization())


  #model.add(Dense(32,activation='relu'))
  #model.add(Dropout(dropout))
  model.add(Bidirectional(keras.layers.LSTM(32)))
  model.add(keras.layers.LeakyReLU())
  model.add(Dropout(dropout))
  model.add(keras.layers.Dense(3,activation='softmax'))
  return model
 


# Cor dos agressores
agressor_color= {'vendedor':'red','direto':'grey','comprador':'green'}
order_colors = {'buy':'green','sell':'red','stop':'yellow'}
corretoras = pd.read_csv('./drive/My Drive/Colab Notebooks/Datasets/trade/CorBov.txt',header=None,names=['codigo','nome'],sep=';') # pegar DF das corretoras