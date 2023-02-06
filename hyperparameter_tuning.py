!pip install wandb
!wandb login

import wandb
from re import X
from keras import layers
from keras import Input
from keras.models import Model
import pandas as pd
import wandb
from keras import optimizers


# Define sweep config
# Slowly adding parameters one by one as they cause an improvement
sweep_config = {
    'method': 'random',
    'name': 'Fragment-Intensity',
    'metric': {'goal': 'maximize', 'name': 'cosine_similarity'},
    'parameters': {
        'N_EPOCHS'  : {'min': 2, 'max': 10},

        'N_FILTERS_1': {'min': 2, 'max': 300},
        'N_FILTERS_2': {'min': 2, 'max': 300},

        'KERNEL_SIZE_1': {'min': 2, 'max': 20},
        'KERNEL_SIZE_2': {'min': 2, 'max': 300},

        'NUM_UNITS_1': {'min': 40, 'max': 300},
        'NUM_UNITS_2': {'min': 40, 'max': 300},

        'BATCH_SIZE': {'value': [32, 64, 128, 256]}
     },

     'metric': {
         'name' : 'cosine_similarity',
         'goal' : 'maximize'
     }
}

sweep_id = wandb.sweep(sweep_config)

def model_tune(config):

  # Adding Input 1 (precursor charge)
  precursor_input = Input(shape=(7,), name='precursor')
  dense = layers.Dense(config['NUM_UNITS_1'], activation='relu')(precursor_input)

  # Adding Input 2 (peptide sequence)
  seq_input = Input(shape=(30,22), name='sequence')
  x = layers.Conv1D(config['N_FILTERS_1'], config['KERNEL_SIZE_1'], activation='relu')(seq_input)
  x = layers.BatchNormalization()(x)
  x = layers.Conv1D(config['N_FILTERS_2'], config['KERNEL_SIZE_2'], activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling1D()(x)
  flatten = layers.Flatten()(x)

  # Concatenate layers
  concat = layers.concatenate([dense, flatten], axis=-1)

  x = layers.Dense(config['NUM_UNITS_2'], activation='relu')(concat)
  output = layers.Dense(56, activation='sigmoid')(x)

  model = Model([precursor_input, seq_input], output, name='baseline')

  return model

  def spectral_angle_loss(y_true, y_pred):
  import keras.backend as k
  import tensorflow
  # Normalize the vectors
  x = k.l2_normalize(y_true, axis=-1)
  y = k.l2_normalize(y_pred, axis=-1)

  # Calculate the dot product between the vectors
  dot_product = k.sum(x * y, axis=-1)

  # Return the spectral angle
  return -(1 - 2 * tensorflow.acos(dot_product) / np.pi )


def train():
  with wandb.init(project='Fragment-Intensity') as run:
    config = wandb.config
    model = model_tune(config)
    model.compile(
        optimizer = optimizers.Adam(),
        loss = spectral_angle_loss,
        metrics = ['cosine_similarity', spectral_angle_loss])
    model.fit(x=[X_s0_train_pre, X_s0_train_seq],
              y=s0_train_int,
              epochs=config['N_EPOCHS'],
              batch_size=config['BATCH_SIZE'],
              validation_data=([X_s0_valid_pre, X_s0_valid_seq], [s0_valid_int]),
              callbacks=[wandb.keras.WandbCallback()])

count = 20
wandb.agent(sweep_id, function=train, count=count)
