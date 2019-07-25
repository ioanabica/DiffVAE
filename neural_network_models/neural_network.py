from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
import numpy as np

from neural_network_models.base_NeuralNetwork import BaseNeuralNetwork


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self,
                 input_size, num_classes, hidden_layers_dim, batch_size, epochs, learning_rate, dropout_probability):
        BaseNeuralNetwork.__init__(self,
                                   input_size=input_size, num_classes=num_classes,
                                   batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
                                   dropout_probability=dropout_probability)

        self.hidden_layers_dim = hidden_layers_dim
        self.build_nn()

    def build_nn(self):
        # Input placeholder
        self.input_placeholder = Input(shape=(self.input_size,))
        nn_layer = self.input_placeholder

        for hidden_dim in self.hidden_layers_dim:
            nn_layer = Dense(hidden_dim, activation='relu')(nn_layer)
            nn_layer = BatchNormalization()(nn_layer)
            nn_layer = Dropout(self.dropout_probability)(nn_layer)

        self.output = Dense(self.num_classes, activation='softmax')(nn_layer)

    def compile_nn(self):
        self.nn_model = Model(inputs=self.input_placeholder, outputs=self.output)

        adam_optimizer = optimizers.Adam(lr=self.learning_rate)
        self.nn_model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer,
                              metrics=['accuracy'])

        print (self.nn_model.summary())

    def train_nn(self, data, labels):
        self.compile_nn()

        # Hot encoding
        labels = labels.astype(int) - 1
        labels = np_utils.to_categorical(labels, self.num_classes)

        # Divide into test and train sets
        perm = np.random.permutation(data.shape[0])
        train_size = int(0.80 * data.shape[0])

        x_train = data[perm[:train_size]]
        y_train = labels[perm[:train_size]]

        x_val = data[perm[train_size:]]
        y_val = labels[perm[train_size:]]

        self.nn_model.fit(x_train, y_train,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          shuffle=True,
                          validation_data=(x_val, y_val))

        score = self.nn_model.evaluate(x_val, y_val, verbose=1)
        print("Validation accuracy:", score[1])

        score = self.nn_model.evaluate(data[perm], labels[perm], verbose=1)
        print("All data accuracy:", score[1])

        self.save_neural_network(neural_network_filename='Saved-Models/NeuralNetworks/simple_nn.h5')
