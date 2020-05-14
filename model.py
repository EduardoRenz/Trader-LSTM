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
    self.checkpoint = ModelCheckpoint(self.save_location, monitor='loss', verbose=1,save_best_only=True, mode='auto', save_freq='epoch')

    self.model = model
    return model   