from funcoes import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU,BatchNormalization,Bidirectional

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.utils.class_weight import compute_class_weight

ORDER_COLORS = {'buy':'green','sell':'red','stop':'yellow'}
METRICS = [
     tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
     tf.keras.metrics.Precision(name='precision'),
     tf.keras.metrics.Recall(name='recall'),
]
SCALE_COLUMNS = ['preco','qtd','vap_comprador','vap_vendedor','vap_5min_comprador','vap_5min_vendedor']

class Trader:

    def __init__(self,save_location):
        self.train_data = None # X e y do treino
        self.train = None # Original DataFrame
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
        self.X_train_df = X_train
        #Reescala os dados
        scaler = StandardScaler()
        X_train_df_scaled = X_train.copy()
        scaler.fit(X_train[SCALE_COLUMNS])
        X_train_df_scaled[SCALE_COLUMNS] = scaler.transform(X_train_df_scaled[SCALE_COLUMNS])
        self.X_train_df_scaled = X_train_df_scaled
        #Oversampling
        X_train_res, y_train_res = oversample(X_train_df_scaled,y_train)
        #Transforma para LSTM
        X_train_lstm,y_train_lstm = create_lstm_dataset(X_train_res,y_train_res,self.time_steps)
        
        train_shape = (None,self.time_steps,X_train_lstm.shape[-1])

        self.train_data = {'X':X_train_lstm, 'y':y_train_lstm}
        self.train_shape = train_shape


    def getClassWeights(self):
        y_integers = np.argmax(self.train_lstm[1], axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))
        return d_class_weights



    def loadTestData(self,path):
        test = loadNegocios(path)
        test = sugestEntrances(test)
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
        X_test_lstm,y_test_lstm = create_lstm_dataset(X_test_scaled,y_test,self.time_steps)

        test_shape = (None,self.time_steps,X_test_lstm.shape[-1])

        self.test_data = (X_test_lstm,y_test_lstm)
        self.test_shape = test_shape



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