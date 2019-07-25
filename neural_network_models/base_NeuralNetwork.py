from keras.models import load_model


def NN_predictions(input_data, neural_network_filename):
    model = load_model(neural_network_filename)
    return model.predict(input_data)


class BaseNeuralNetwork():

    def __init__(self, input_size, num_classes, batch_size, epochs, learning_rate, dropout_probability):
        self.input_size = input_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_probability = dropout_probability

    def save_neural_network(self, neural_network_filename):
        self.nn_model.save(neural_network_filename)

    def restore_neural_network(self, neural_network_filename):
        self.nn_model = load_model(neural_network_filename)

    def predict(self, input_data, neural_network_filename):
        self.restore_neural_network(neural_network_filename)
        return self.nn_model.predict(input_data)
