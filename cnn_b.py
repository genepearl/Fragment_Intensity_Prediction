import tensorflow as tf
import numpy as np
from keras import layers
from keras import Input
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers

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

def create_model():

  # Adding Input A (precursor charge)
  precursor_input = Input(shape=(7,), name='precursor')
  dense = layers.Dense(7, activation='relu')(precursor_input)

  # Adding Input B (peptide sequence)
  seq_input = Input(shape=(30,22), name='sequence')
  x = layers.Conv1D(64, 7, activation='relu')(seq_input)
  x = layers.BatchNormalization()(x)
  x = layers.Conv1D(256, 5, activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling1D()(x)
  flatten = layers.Flatten()(x)

  # Concatenate layers
  concat = layers.concatenate([dense, flatten], axis=-1)

  x = layers.Dense(128, activation='relu')(concat)
  output = layers.Dense(56, activation='sigmoid')(x)

  model = Model([precursor_input, seq_input], output, name='baseline')

  return model

def compile_model(model):
  model.compile(
      optimizer = optimizers.Adam(),
      loss = spectral_angle_loss,
      metrics = ['cosine_similarity', spectral_angle_loss])

def load_and_fit (model, num_epochs=5, batch_size=128)
  for i in range(0, 5):
      train_int = np.load(f"/content/drive/MyDrive/splits_mod/split{i}/s{i}_train_int.npy")
      train_pre = np.load(f"/content/drive/MyDrive/splits_mod/split{i}/s{i}_train_pre.npy")
      train_seq = np.load(f"/content/drive/MyDrive/splits_mod/split{i}/s{i}_train_seq.npy")
      valid_int = np.load(f"/content/drive/MyDrive/splits_mod/split{i}/s{i}_valid_int.npy")
      valid_pre = np.load(f"/content/drive/MyDrive/splits_mod/split{i}/s{i}_valid_pre.npy")
      valid_seq = np.load(f"/content/drive/MyDrive/splits_mod/split{i}/s{i}_valid_seq.npy")

      # One-hot encode the peptide sequences and the intensities
      train_seq = tf.one_hot(train_seq, 22)
      train_pre = tf.one_hot(train_pre, 7)
      valid_seq = tf.one_hot(valid_seq, 22)
      valid_pre = tf.one_hot(valid_pre, 7)

      # Train the model on the current cross-validation set
      model.fit(x=[train_pre, train_seq],
                y=train_int,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=([valid_pre, valid_seq], [valid_int]))

    # Free the memory by deleting the one-hot encoded data
    del train_seq, train_pre, train_int, valid_seq, valid_pre, valid_int
    tf.keras.backend.clear_session()
  return model

def load_and_encode_holdout('):
  hold_pre = np.load('/content/drive/MyDrive/holdout_mod/test_pre.npy')
  hold_seq = np.load('/content/drive/MyDrive/holdout_mod/test_seq.npy')
  hold_int = np.load('/content/drive/MyDrive/holdout_mod/test_int.npy')

  # One-Hot Encoding of Holdout Sequence
  X_hold_seq = tf.one_hot(hold_seq, depth=22)

  # One-Hot Encoding of Holdout  Precursor Charge
  X_hold_pre = tf.one_hot(hold_pre, depth=7)
  return X_hol_seq, X_hold_pre
