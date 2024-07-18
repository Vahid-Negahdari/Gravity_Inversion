import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle
import matplotlib
import matplotlib.pyplot as plt

#path = Path('/home/cvl/Pycharm/Gravity_Density_Inversion')
path = Path('G:\projet\Gravity Density Inversion')
#########################################################
# Define Hyperparameter
#########################################################
train_epochs = 35
batch_size   = 20
BIGG_BATCH   = 27000
num_batch    = int(BIGG_BATCH/batch_size)
lr           = 0.0005
n            = 51
k            = 100
t            = 102
semi         = int(np.ceil(n/8)*np.ceil(n/8)*128)
#########################################################
# Import Data
#########################################################
A       = tf.constant(np.load(path / ('A.npy'), allow_pickle=True))
Density = np.load(path / ('Density_G.npy'), allow_pickle=True).astype('float32')
#Density = (Density-np.min(Density))/(np.max(Density)-np.min(Density))
Gravity = np.load(path / ('Gravity.npy'), allow_pickle=True)

Density = tf.expand_dims(tf.constant(Density[t]) , 1)
Gravity = tf.expand_dims(tf.constant(Gravity[t]) , 1)

#########################################################
# Define Some Functions
#########################################################
def conv(input, w, stride, dimention):
    if dimention == 1 :
       y = tf.nn.conv1d(input=input, filters=w, stride=stride,padding='SAME')
    else :
       y = tf.nn.conv2d(input=input, filters=w, strides=stride, padding='SAME')
    y = tf.nn.relu(y)
    return y


def deconv(input, w, strides, output, dimention):
    if dimention == 1:
       y = tf.nn.conv1d_transpose(input=input, filters=w, strides=strides, padding='SAME', output_shape=output)
    else:
       y = tf.nn.conv2d_transpose(input=input, filters=w, strides=strides, padding='SAME', output_shape=output)
    y = tf.nn.relu(y)
    return y


def fullyConnected_layer(input,w,b):
  y = tf.matmul(input,w) + b
  return y

#########################################################
# Define Weights
#########################################################
def Load_WEIGHTS():
    open_file = open('Weights_Gen1.pkl', "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

GenV = Load_WEIGHTS()
for i in range(len(GenV)):
    GenV[i] = tf.constant(GenV[i].numpy())

Initial = np.random.normal(0, 1, [1, k]).astype('float32')

def get_tfVariable(name):
    return tf.Variable(Initial, name=name, trainable=True, dtype=tf.float32)

Weights = []
Weights = Weights + [get_tfVariable('W5')]

########################################################
# Define Model
########################################################
def Generator():
    C = fullyConnected_layer(Weights[0], GenV[0], GenV[1])
    C = tf.reshape(C, [C.shape[0], int(np.ceil(n/8)), int(np.ceil(n/8)), 128])
    C = deconv(C, GenV[2], 2, [C.shape[0], int(np.ceil(n/4)), int(np.ceil(n/4)), 64], 2)
    C = deconv(C, GenV[3], 2, [C.shape[0], int(np.ceil(n/2)), int(np.ceil(n/2)), 32], 2)
    C = deconv(C, GenV[4], 2, [C.shape[0], n, n, 1], 2)
    C = tf.reshape(C,[n*n,1])
    return C


#########################################################
# Define Loss Function
#########################################################
def Generator_loss(Fake):
#     Loss = tf.reduce_mean(tf.abs(tf.matmul(A,Fake) - Gravity))
     Loss = tf.reduce_mean(tf.abs(Fake - Density))
     return  Loss


#######################################################
#######################################################
def train_step(lr1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr1)
    with tf.GradientTape() as tape:
        Fake   = Generator()
        Loss   = Generator_loss(Fake)
        grad = tape.gradient(Loss, Weights )
        optimizer.apply_gradients(zip(grad, Weights ))
        return Loss


################################################################
# Training Process
################################################################
for epoch in range(train_epochs):
      avg_Loss = 0
      if np.mod(epoch,3)==0:
         lr=lr/2

      for s in range(num_batch):
          batch_u         = Density  [s * batch_size  : (s + 1) * batch_size ]
          loss  = train_step(lr)
          avg_Loss += loss / num_batch

      print("--- On epoch {} ---".format(epoch))
      tf.print(" ---Loss1:---", avg_Loss)
      print("\n")



B = Generator().numpy().reshape([n,n])
C = Density.numpy().reshape([n,n])


# ex=np.load(r'C:\Users\Vahid\Desktop\k100b20.npy')
# for i in range(200):
#         plt.imshow(ex[i])
#         plt.savefig(str(i)  +'.png')
#         plt.close()
