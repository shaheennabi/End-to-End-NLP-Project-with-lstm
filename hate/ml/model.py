from hate.entity.config_entity import ModelTrainerConfig
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.optimizers import RMSprop

class ModelArchitecture:

    def __init__(self, config: ModelTrainerConfig):
        # Initialize with the config object and set required parameters
        self.MAX_WORDS = config.MAX_WORDS
        self.MAX_LEN = config.MAX_LEN
        self.LOSS = config.LOSS
        self.METRICS = config.METRICS
        self.ACTIVATION = config.ACTIVATION

    def get_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.MAX_WORDS, output_dim=100))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation=self.ACTIVATION))

        # Compile the model
        model.compile(
            optimizer=RMSprop(),
            loss=self.LOSS,
            metrics=self.METRICS
        )

        # Force the model to build by passing a dummy input with the correct shape
        model.build(input_shape=(None, self.MAX_LEN))

        # Display the model summary
        model.summary()

        return model
