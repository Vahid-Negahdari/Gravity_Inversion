import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
import numpy as np
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import matplotlib.pyplot as plt
import time

##########################################################################################################################
# Hyperparameters
##########################################################################################################################

epochs     = 3000
n          = 51
train_size = 27000
batch_size = 30
num_batch  = int(train_size/batch_size)
latent_dim = 50
lr         = 1.0*1e-3



##########################################################################################################################
# Load gravity and density datasets
##########################################################################################################################

path = Path.cwd() /('Dataset')

if not path.exists():
    print("Dataset not found. Downloading...")
    url = "https://github.com/Vahid-Negahdari/Gravity_Inversion/releases/download/v1.0.0/Gravity_Inversion.zip"
    try:
        http_response = urlopen(url)
        with ZipFile(BytesIO(http_response.read())) as archive:
            archive.extractall(path)
        print("Dataset downloaded and extracted to:", path)
    except Exception as e:
        print("Error while downloading dataset:", e)
else:
    print("Dataset already exists at:", path)





Density_Train = np.load(path / ('Density_Train.npy'), allow_pickle=True)
Density_Train = np.reshape(Density_Train,[27000,n,n,1] )/12
Density_Test = np.load(path / ('Density_Test.npy'), allow_pickle=True)
Density_Test = np.reshape(Density_Test,[1000,n,n,1] )/12


Gravity_Train = np.load(path / ('Gravity_Train.npy'), allow_pickle=True)
Gravity_Train = np.reshape(Gravity_Train,[27000,3*n,2] )
Gravity_Test = np.load(path / ('Gravity_Test.npy'), allow_pickle=True)
Gravity_Test = np.reshape(Gravity_Test,[1000,3*n,2] )


train_dataset = tf.data.Dataset.from_tensor_slices( (Gravity_Train , Density_Train) ).shuffle(train_size).batch(batch_size)


##########################################################################################################################
# Define Model
##########################################################################################################################

Forward = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(3*n, 2)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, activation=LeakyReLU()),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation=LeakyReLU()),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation=LeakyReLU()),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(250, activation=LeakyReLU() ),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(500, activation=LeakyReLU()),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1000, activation=LeakyReLU()),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1500, activation=LeakyReLU()),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(2601, activation='sigmoid'),
    tf.keras.layers.Reshape((51, 51, 1))])


##########################################################################################################################
# Optimization
##########################################################################################################################

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
              initial_learning_rate=lr,  decay_steps=num_batch*100,  decay_rate=0.8, staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule, weight_decay=1e-4)





def compute_loss(x, y):
  preds = Forward(x, training=True)
  loss  = (12**2)*tf.reduce_mean( tf.square(preds-y))
  return loss


@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
       loss = compute_loss(x, y)
  gradients = tape.gradient(loss, Forward.trainable_variables )
  optimizer.apply_gradients(zip(gradients, Forward.trainable_variables))
  return loss



##########################################################################################################################
# Training loop
##########################################################################################################################
def Test_Score():
    preds = Forward(Gravity_Test, training=False)
    loss = (12**2)*tf.reduce_mean(tf.square(preds - Density_Test))
#    current_lr = optimizer.learning_rate(optimizer.iterations).numpy()
    print('-------------------------------------------------')
    print('Test_Loss: {}'.format(loss))
#    print(f"  Learning Rate: {current_lr:.5f}")
    print('-------------------------------------------------')


for epoch in range(0, epochs ):
    if np.mod(epoch,10)==0 :
        Test_Score()

    start_time = time.time()
    avg  = 0
    for batch_x , batch_y in train_dataset:
        loss  = train_step(batch_x, batch_y)
        avg  += (loss  / (num_batch))

    end_time = time.time()
    print('Epoch: {},  loss: {}, time elapse for current epoch: {}'.format(epoch, avg, end_time - start_time))



