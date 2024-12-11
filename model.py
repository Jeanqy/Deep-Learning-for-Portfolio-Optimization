import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense, Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Bidirectional
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K

class Model_LSTM:
    def __init__(self, loss = "paper", reg = False, structure_change = False):
        self.data = None
        self.model = None
        self.loss = loss
        self.reg = reg
        self.structure_change = structure_change
        

    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''

        kernel_regularizer = None
        if self.reg:
            kernel_regularizer = tf.keras.regularizers.l2(10)

        if self.structure_change == "autoencoder":
            input_encoder = Input(shape=input_shape)
            # encoding_dim is the desired dimensionality of the encoded representation
            n_row, n_col = input_shape
            encoding_dim = 0.5 * n_col
            encoded = Dense(encoding_dim, activation='relu')(input_encoder)  
            encoder_model = Model(input_encoder, encoded)
            model = Sequential([
                encoder_model,
                LSTM(64),
                Flatten(),
                Dense(outputs, activation='softmax', kernel_regularizer = kernel_regularizer)
            ])

        elif self.structure_change == "SAE_CNN_LSTM":
            # Define the Stacked Autoencoder
            input_encoder = Input(shape=input_shape)
            n_row, n_col = input_shape
            encoding_dim_1 = int(0.75 * n_col)  # Adjust this as needed
            encoding_dim_2 = 4  # Setting this to 4 to match your error description
           
            # Encoding layers (2 layers as an example for "stacked")
            encoded_1 = Dense(encoding_dim_1, activation='relu')(input_encoder)
            encoded_2 = Dense(encoding_dim_2, activation='relu')(encoded_1)
           
            encoder_model = Model(input_encoder, encoded_2)  # This model compresses the input
           
            # Define the combined model
            input_layer = Input(shape=input_shape)
           
            # Pass input through the encoder
            encoded_input = encoder_model(input_layer)
           
            # CNN component
            x = Conv1D(32, 3, activation='relu')(encoded_input)
            x = MaxPooling1D(2)(x)
           
            # LSTM and Dense layers
            x = LSTM(64)(x)
            x = Flatten()(x)
            output_layer = Dense(outputs, activation='softmax', kernel_regularizer=kernel_regularizer)(x)
           
            model = Model(input_layer, output_layer)

        elif self.structure_change == "SAE_3CNN_LSTM":
            input_encoder = Input(shape=input_shape)
            n_row, n_col = input_shape
            encoding_dim_1 = int(0.75 * n_col)  # Adjust this as needed
            encoding_dim_2 = 4  # Setting this to 4 to match your error description
           
            # Encoding layers (2 layers as an example for "stacked")
            encoded_1 = Dense(encoding_dim_1, activation='relu')(input_encoder)
            encoded_2 = Dense(encoding_dim_2, activation='relu')(encoded_1)
           
            encoder_model = Model(input_encoder, encoded_2)  # This model compresses the input
           
            # Define the combined model
            input_layer = Input(shape=input_shape)
           
            # Pass input through the encoder
            encoded_input = encoder_model(input_layer)
           
            # Enhanced CNN component
            x = Conv1D(32, 3, activation='relu')(encoded_input)
            x = MaxPooling1D(2)(x)
           
            x = Conv1D(64, 3, activation='relu')(x)  # Additional Conv layer with 64 filters
            x = MaxPooling1D(2)(x)  # Additional MaxPooling layer
           
            x = Conv1D(128, 3, activation='relu')(x)  # Additional Conv layer with 128 filters
            x = MaxPooling1D(2)(x)  # Additional MaxPooling layer
           
            # LSTM and Dense layers
            x = LSTM(64)(x)
            x = Flatten()(x)
            output_layer = Dense(outputs, activation='softmax', kernel_regularizer=kernel_regularizer)(x)
           
            model = Model(input_layer, output_layer)

        elif self.structure_change == "Double_LSTM":
            model = Sequential([
                # First LSTM Layer with Dropout and Recurrent Dropout
                LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape),
                BatchNormalization(),  # Batch Normalization after the first LSTM layer

                # Second LSTM Layer with Dropout and Recurrent Dropout
                LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
                BatchNormalization(),  # Batch Normalization after the second LSTM layer

                Flatten(),
                Dense(outputs, activation='softmax', kernel_regularizer=kernel_regularizer)
            ])

        elif self.structure_change == "BiLSTM":
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True,dropout=0.2, input_shape=input_shape)),
                Flatten(),
                Dense(outputs, activation='softmax', kernel_regularizer=kernel_regularizer)
            ])

        else:
            model = Sequential([
                LSTM(64, input_shape=input_shape),
                Flatten(),
                Dense(outputs, activation='softmax', kernel_regularizer = kernel_regularizer)
            ])

            
        @tf.autograph.experimental.do_not_convert   
        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula
            
            # mean = tf.reduce_mean(portfolio_returns, axis=0)
            # stddev = tf.math.reduce_std(portfolio_returns, axis=0)
            # portfolio_returns = (portfolio_returns - mean) / stddev
            
            if self.loss == "paper":
                loss = K.mean(portfolio_returns) / K.std(portfolio_returns)
            elif self.loss == "return":
                loss = K.mean(portfolio_returns)
            elif self.loss == "convex":
                loss = K.mean(portfolio_returns) - K.std(portfolio_returns)
            elif self.loss == "sortino":
                loss = K.mean(portfolio_returns) / K.std(portfolio_returns[portfolio_returns<0])
            elif self.loss == "sortino_convex":
                loss = K.mean(portfolio_returns) - K.std(portfolio_returns[portfolio_returns<0])
            return -loss
        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    
    def get_allocations(self, data: pd.DataFrame):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns
        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        
        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, len(data.columns))
        
        fit_predict_data = data_w_ret[np.newaxis,:]        
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=20, shuffle=False,
                       verbose=0
                      )
        return self.model.predict(fit_predict_data)[0]
    
    