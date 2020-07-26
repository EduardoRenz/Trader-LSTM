from funcoes import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU,BatchNormalization,Bidirectional

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.utils.class_weight import compute_class_weight

ORDER_COLORS = {'buy':'green','sell':'red','stop':'yellow','acao_sell':'red','acao_buy':'green'}
METRICS = [
     tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
     tf.keras.metrics.Precision(name='precision'),
     tf.keras.metrics.Recall(name='recall'),
]
SCALE_COLUMNS = ['preco','qtd','vap_comprador','vap_vendedor','vap_5min_comprador','vap_5min_vendedor']

class Trader:

    def __init__(self,save_location='last_weight'):
        self.train_data = None # X e y do treino
        self.train = None # Original DataFrame
        self.scaler = None
        self.X_train_df = None # DataFrame after dummies columns
        self.X_train_df_scaled = None
        self.train_lstm = None # LSTM data for training
        self.train_shape = None
        self.test_data = None # X e y do teste
        self.test = None # Original DataFrame
        self.test_df = None # DataFrame after dummies columns
        self.test_shape = None
        self.save_location = save_location
        self.checkpoint = None # Callback from keras to automatic save the model weights
        self.time_steps = 50
        self.model = None



    #Retorna dados de treino e teste quando for no mesmo dia
    def getTrainTest(self,negocios,test_percent=60):
        #One hot dos dados
        df = negociosWithDummies(negocios)

        # Obter colunas de treino e teste
        y_columns = pd.get_dummies(negocios[['acao']]).sort_index(axis=1).columns
        X_columns =list(filter(lambda x : x not in y_columns and x not in ['preco'],df.columns))

        #Train e test
        train_data = df.iloc[: int((len(df) /100) * test_percent)]
        test_data = df.iloc[int((len(df) /100) * test_percent):]

        X_train = train_data[X_columns]
        y_train = train_data[y_columns].sort_index(axis=1)

        X_test = test_data[X_columns]
        y_test= test_data[y_columns].sort_index(axis=1)

        return X_columns, y_columns,train_data,test_data, X_train, y_train,X_test,y_test

    # Transforma os dados de treino para escala
    def scaleTrainTest(self,X_train,X_test,scaler_columns=['qtd','vap_5min_comprador','vap_5min_vendedor','vap_15min_comprador','vap_15min_vendedor']):
        scaler = StandardScaler()

        X_train_scaled = X_train.copy()
        X_test_scaled =X_test.copy()

        scaler.fit(X_train_scaled[scaler_columns])
        X_train_scaled[scaler_columns] = scaler.transform(X_train_scaled[scaler_columns])
        X_test_scaled[scaler_columns] = scaler.transform(X_test_scaled[scaler_columns])
        return scaler, X_train_scaled,X_test_scaled

    #Abstrai o carregamento de train e test para somente train data
    def loadTrainData(self,path):
        train = loadNegocios(path)
        train = sugestEntrances(train)
        self.train = train
        X_columns, y_columns, train_data,test_data, X_train, y_train, X_test, y_test = self.getTrainTest(train,99)

        self.X_train_df = X_train
        self.y_train = y_train
        # #Reescala os dados
        scaler, X_train_df_scaled,X_test_scaled = self.scaleTrainTest(X_train,X_test)
        self.scaler = scaler
        self.X_train_df_scaled = X_train_df_scaled

        #Transforma para LSTM
        X_train_lstm,y_train_lstm = create_lstm_dataset(X_train_df_scaled,y_train,self.time_steps)
        self.X_train_lstm = X_train_lstm
        self.y_train_lstm = y_train_lstm
        
        train_shape = (None,self.time_steps,X_train_lstm.shape[-1])

        self.train_data = {'X':X_train_lstm, 'y':y_train_lstm}
        self.train_shape = train_shape



    def loadTestData(self,path):
        test = loadNegocios(path)
        test = sugestEntrances(test)
        self.test = test
        X_columns, y_columns, train_data,test_data, X_train, y_train, X_test, y_test = self.getTrainTest(test,99)

        self.X_test_df = X_test
        self.y_test = y_test
        # #Reescala os dados
        scaler, X_train_df_scaled,X_test_df_scaled = self.scaleTrainTest(X_test,X_test)
        self.scaler = scaler
        self.X_test_df_scaled = X_test_df_scaled

        #Transforma para LSTM
        X_test_lstm,y_test_lstm = create_lstm_dataset(X_test_df_scaled,y_test,self.time_steps)
        
        train_shape = (None,self.time_steps,X_test_lstm.shape[-1])

        self.test_data = {'X':X_test_lstm, 'y':y_test_lstm}



    def getClassWeights(self):
        y_integers = np.argmax(self.y_train.values, axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))
        return d_class_weights



    def plotGraph(self,df):
        fig = buildMainGraph(df)
        #Ações que devem ser tomadas (comprar e vender)
        for (acao,v) in df[df != 'do nothing'].groupby('acao'):
            if(acao == 'do nothing'):
                continue
            fig.add_trace(go.Scatter(x=v.index,y=v.preco,mode='markers',name=acao, marker={'color':ORDER_COLORS[acao] }),row=1,col=2)

        fig.update_layout(barmode='stack',margin=dict(r=10, t=10, b=10, l=10),width=1200)
        fig.show()

    def plotPredictions(self,prediction_df):
        n_buys = len(prediction_df.query('acao_buy >=0.7'))
        n_sells = len(prediction_df.query('acao_sell >=0.7'))
        #PLotar o Grafico de treino e teste
        fig = make_subplots(shared_yaxes=True,shared_xaxes=True)
        #Add a linha do preço
        fig.add_trace(go.Scatter( x=self.test_df.index, y=self.test_df.preco, name="Preço"  ))

        # Predictions
        for v in prediction_df.sort_values(by='acao_buy',ascending=False).head(n_buys).itertuples():
            fig.add_trace(go.Scatter(x=[v.Index],y=self.test_df[self.test_df.index == v.Index].preco,mode='markers',text=round(v.acao_buy*100,2),legendgroup='buys', name='predicted buy',marker={'color':'#a0e69c'}))

        for v in prediction_df.sort_values(by='acao_sell',ascending=False).head(n_sells).itertuples():
            fig.add_trace(go.Scatter(x=[v.Index],y=self.test_df[self.test_df.index == v.Index].preco,mode='markers',name='predicted sell',text=round(v.acao_sell*100,2),marker={'color':'#e03884'}))

        fig.update_layout(barmode='stack',margin=dict(r=10, t=10, b=10, l=10),width=1200)
        fig.show()

    def buildModel(self,input_shape):
        dropout = 0.3 # dropout para cada lstm

        model = keras.Sequential()
        #model.add(keras.layers.LSTM(8, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=input_shape)))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(256,return_sequences=True)))
        model.add(LeakyReLU())
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

        # model.add(LSTM(128,return_sequences=True))
        #model.add(LeakyReLU())
        # model.add(Dropout(dropout))
        # model.add(BatchNormalization())


        #model.add(Dense(128,activation='relu'))
        #model.add(Dropout(dropout))
        model.add(keras.layers.LSTM(128))
        model.add(keras.layers.LeakyReLU())
        model.add(Dropout(dropout))
        model.add(keras.layers.Dense(3,activation='softmax'))
        self.model = model
        return model

    def fit(self,load_weights=False):
        optimizer = keras.optimizers.Adam(lr=0.001)
        #Varias metricas para testar durante o treino
        METRICS = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
        d_class_weights = self.getClassWeights()



        self.model.compile(loss=keras.losses.CategoricalCrossentropy(),optimizer=optimizer,metrics=METRICS,sample_weight_mode="temporal")
        checkpoint = ModelCheckpoint(self.save_location, monitor='loss', verbose=1,save_best_only=True, mode='auto', save_freq="epoch")


        self.model.build(self.train_shape)
        if load_weights:
            self.model.load_weights(self.saved_weights)
        #%tensorboard --logdir logs
        #,class_weight=d_class_weights
        results = self.model.fit(self.X_train_lstm, self.y_train_lstm, epochs=200, batch_size=100,callbacks=[checkpoint],class_weight=d_class_weights, verbose=1,shuffle=False)
    