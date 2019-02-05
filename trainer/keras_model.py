from tensorflow.python import keras
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import backend as K


# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb
class Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.w_regularizer = None
        self.b_regularizer = None

        self.w_constraint = None
        self.b_constraint = None
        self.w = None
        self.b = None
        self.bias = True
        self.output_dim = output_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        # self.w = self.add_weight(shape=(input_shape[-1],),
        self.w = self.add_weight(shape=(19,),
                                 initializer=self.init,
                                 name='{}_w'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        # self.features_dim = input_shape[-1]
        self.features_dim = 19

        if self.bias:
            # self.b = self.add_weight(shape=(input_shape[1],),
            self.b = self.add_weight(shape=(160,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        super(Attention, self).build(input_shape=input_shape)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(Attention, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.output_dim

        eij = tf.keras.backend.reshape(tf.keras.backend.dot(tf.keras.backend.reshape(x, (-1, features_dim)),
                                       tf.keras.backend.reshape(self.w, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = tf.keras.backend.tanh(eij)

        a = tf.keras.backend.exp(eij)

        if mask is not None:
            a *= tf.keras.backend.cast(mask, tf.keras.backend.floatx())

        a /= tf.keras.backend.cast(tf.keras.backend.sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.keras.backend.floatx())

        a = tf.keras.backend.expand_dims(a)
        weighted_input = x * a
        return tf.keras.backend.sum(weighted_input, axis=1)


# It is the official metric used in this competition
# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)
def matthews_correlation(y_true, y_pred):
    """Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    """
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = tf.keras.backend.sum(y_pos * y_pred_pos)
    tn = tf.keras.backend.sum(y_neg * y_pred_neg)

    fp = tf.keras.backend.sum(y_neg * y_pred_pos)
    fn = tf.keras.backend.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = tf.keras.backend.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + tf.keras.backend.epsilon())


# This is NN LSTM Model creation
def model_lstm(input_shape):
    # The shape was explained above, must have this order
    inp = tf.keras.Input(shape=(input_shape[1], input_shape[2]))
    # This is the LSTM layer
    # Bidirecional implies that the 160 chunks are calculated in both ways, 0 to 159 and 159 to zero
    # although it appear that just 0 to 159 way matter, I have tested with and without, and tha later worked best
    # 128 and 64 are the number of cells used, too many can overfit and too few can underfit
    x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(128, return_sequences=True), input_shape=(160, 2))(inp)
    # The second LSTM can give more fire power to the model, but can overfit it too
    x = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(64, return_sequences=True))(x)
    # Attention is a new technology that can be applied to a Recurrent NN
    # to give more meanings to a signal found in the middle
    # of the data, it helps more in longs chains of data. A normal RNN give all the responsibility of detect the signal
    # to the last cell. Google RNN Attention for more information :)
    att = Attention(input_shape[1])
    x = att(x)
    # A intermediate full connected (Dense) can help to deal with non-linear outputs
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # A binary classification as this must finish with shape (1,)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inp, outputs=x)
    # Pay attention in the addition of matthews_correlation metric in the compilation, it is a success factor key
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])

    return model


def get_estimator(model, model_dir, run_config):
    return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir, config=run_config)
