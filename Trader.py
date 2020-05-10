from funcoes import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU,BatchNormalization,Bidirectional

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ORDER_COLORS = {'buy':'green','sell':'red','stop':'yellow'}

METRICS = [
     tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
     tf.keras.metrics.Precision(name='precision'),
     tf.keras.metrics.Recall(name='recall'),
]



class Trader:

    def __init__(self,save_location):
        self.train_data = None # X e y do treino
        self.train = None # Original DataFrame
        self.train_df = None # DataFrame depois de aplicado dummies
        self.train_lstm = None # LSTM dos dados de treino
        self.train_shape = None
        self.test_data = None # X e y do teste
        self.test = None # Original DataFrame
        self.test_df = None # DataFrame depois de aplicado dummies
        self.test_shape = None
        self.model = None
        self.save_location = save_location


    def buildModel(self,input_shape):
        optimizer = keras.optimizers.Adam(lr=0.001)
        dropout = 0.2 # dropout para cada lstm

        model = keras.Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(128,return_sequences=True)))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

        model.add(Bidirectional(keras.layers.LSTM(32)))
        model.add(keras.layers.LeakyReLU())
        model.add(Dropout(dropout))
        model.add(keras.layers.Dense(3,activation='softmax'))

        model.compile(loss=keras.losses.CategoricalCrossentropy(),optimizer=optimizer,metrics=METRICS)
        checkpoint = ModelCheckpoint(self.save_location, monitor='loss', verbose=1,save_best_only=True, mode='auto', save_freq=1000)

        self.model = model
        return model   
        

    def loadTrainData(self,path):
        train = loadNegocios(path)
        train = sugestEntrances(train)
        self.train = train
        # Gera df agora com dummies
        train_w_targets = negociosWithDummies(train).copy()
        # Obter colunas de treino e teste
        y_columns = pd.get_dummies(train[['acao']]).sort_index(axis=1).columns
        X_columns =list(filter(lambda x : x not in y_columns,train_w_targets.columns))
        # Train e train
        X_train = train_w_targets[X_columns].copy()
        y_train= train_w_targets[y_columns].sort_index(axis=1).copy()
        self.train_df = X_train
        #Reescala os dados
        scaler = StandardScaler()
        scaler.fit(X_train[['qtd','preco']])
        X_train_scaled = X_train.copy()
        X_train_scaled[['qtd','preco']] = scaler.transform(X_train[['qtd','preco']])
        #Oversampling
        X_train_res, y_train_res = oversample(X_train_scaled,y_train)
        #Transforma para LSTM
        time_steps = 10
        X_train_lstm,y_train_lstm = create_lstm_dataset(X_train_res,y_train_res,time_steps)
        
        self.train_lstm = (X_train_lstm,y_train_lstm)

        train_shape = (None,time_steps,X_train_lstm.shape[-1])

        self.train_data = (X_train_lstm,y_train_lstm)
        self.train_shape = train_shape


    def fit(self,epochs=200,build_model=True):
        if build_model:
            self.buildModel(self.train_shape)
        results = self.model.fit(self.train_lstm[0], self.train_lstm[1], epochs=epochs, batch_size=100,callbacks=[checkpoint], verbose=1)



    def loadTestData(self,path):
        test = loadNegocios(path)
        self.test = test
        # Gera df agora com dummies
        test_w_targets = negociosWithDummies(test).copy()
        # Obter colunas de treino e teste
        y_columns = pd.get_dummies(test[['acao']]).sort_index(axis=1).columns
        X_columns =list(filter(lambda x : x not in y_columns,test_w_targets.columns))
        # Train e test
        X_test = test_w_targets[X_columns].copy()
        self.test_df = X_test
        y_test= test_w_targets[y_columns].sort_index(axis=1).copy()
        #Reescala os dados
        scaler = StandardScaler()
        scaler.fit(X_test[['qtd','preco']])
        X_test_scaled = X_test.copy()
        X_test_scaled[['qtd','preco']] = scaler.transform(X_test[['qtd','preco']])
        #Transforma para LSTM
        time_steps = 50
        X_test_lstm,y_test_lstm = create_lstm_dataset(X_test_scaled,y_test,time_steps)

        test_shape = (None,time_steps,X_test_lstm.shape[-1])

        self.test_data = (X_test_lstm,y_test_lstm)
        self.test_shape = test_shape





    def showGraph(self,df):
        fig = buildMainGraph(df)
        #Ações que devem ser tomadas (comprar e vender)
        for (acao,v) in df[df != 'do nothing'].groupby('acao'):
            if(acao == 'do nothing'):
                continue
            fig.add_trace(go.Scatter(x=v.index,y=v.preco,mode='markers',name=acao, marker={'color':ORDER_COLORS[acao] }),row=1,col=2)

        fig.update_layout(barmode='stack',margin=dict(r=10, t=10, b=10, l=10),width=1200)
        fig.show()