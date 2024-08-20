import LSTM

class Environment:
    def __init__(self, config):
        self.model_type = config['architecture']
        self.hidden_dim = config["hidden_dim"]
        self.dropout = config['dropout']
        self.n_layers = config["layers"]
        self.total_epochs = config['total_epochs']
        self.env_learning_rate = config["ENV_LEARNING_RATE"]
        self.input_shape = config['input_shape']
        self.dense_units=config['dense_units']
        self.optimizer=config['optimizer']
        self.config = config
        self.model = None

        if self.model_type == 'LSTM':
            self.model = LSTM.build_and_compile_model(
                input_shape=self.input_shape,
                num_layers=self.n_layers,
                hidden_dims=self.hidden_dim,
                dropout_rate=self.dropout,
                dense_units=self.dense_units,
                lr=self.env_learning_rate,
                optimizer=self.optimizer,
                loss='mean_absolute_error'
            )
