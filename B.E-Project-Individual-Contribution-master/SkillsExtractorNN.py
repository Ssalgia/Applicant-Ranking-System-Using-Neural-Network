class SkillsExtractorNN:

    def __init__(self, word_features_dim, dense_features_dim):

        lstm_input_phrase = keras.layers.Input(shape=(None, word_features_dim))
        lstm_input_cont = keras.layers.Input(shape=(None, word_features_dim))
        dense_input = keras.layers.Input(shape=(dense_features_dim,))

        lstm_emb_phrase = keras.layers.LSTM(256)(lstm_input_phrase)
        lstm_emb_phrase = keras.layers.Dense(128, activation='relu')(lstm_emb_phrase)

        lstm_emb_cont = keras.layers.LSTM(256)(lstm_input_cont)
        lstm_emb_cont = keras.layers.Dense(128, activation='relu')(lstm_emb_cont)

        dense_emb = keras.layers.Dense(512, activation='relu')(dense_input)
        dense_emb = keras.layers.Dense(256, activation='relu')(dense_emb)

        x = keras.layers.concatenate([lstm_emb_phrase, lstm_emb_cont, dense_emb])
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)

        main_output = keras.layers.Dense(2, activation='softplus')(x)

        self.model = keras.models.Model(inputs=[lstm_input_phrase, lstm_input_cont, dense_input],
                                        outputs=main_output)

        optimizer = keras.optimizers.Adam(lr=1)

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')



    def fit(self, x_lstm_phrase, x_lstm_context, x_dense, y,
            val_split=0.25, patience=5, max_epochs=1000, batch_size=32):

        x_lstm_phrase_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_phrase)
        x_lstm_context_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_context)

        y_onehot = onehot_transform(y)

        self.model.fit([x_lstm_phrase_seq, x_lstm_context_seq, x_dense],
                       y_onehot,
                       batch_size=batch_size,
                       pochs=max_epochs,
                       validation_split=val_split,
                       callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)])


    def predict(self, x_lstm_phrase, x_lstm_context, x_dense):

        x_lstm_phrase_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_phrase)
        x_lstm_context_seq = keras.preprocessing.sequence.pad_sequences(x_lstm_context)

        y = self.model.predict([x_lstm_phrase_seq, x_lstm_context_seq, x_dense])

        return y


def onehot_transform(y):

    onehot_y = []

    for numb in y:
        onehot_arr = np.zeros(2)
        onehot_arr[numb] = 1
        onehot_y.append(np.array(onehot_arr))

    return np.array(onehot_y)
