import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


train_epochs    = 20
batch_size      = 25
BIGG_BATCH      = 27000
num_BIGG_BATCH  = 1
num_batch       = int(BIGG_BATCH/batch_size)
n               = 51
N               = n**2
semi            = int(np.ceil(N/4))*128
Z               = 500
latent_space    = 15
lr             = 0.01
k              = 25


#path = Path('/home/cvl/Pycharm/Gravity_Density_Inversion')
path    = Path('G:\projet\Gravity Density Inversion')
A       = tf.constant(np.load(path / ('A.npy'), allow_pickle=True))
Density = np.load(path / ('Density.npy'), allow_pickle=True)
Gravity = np.load(path / ('Gravity.npy'), allow_pickle=True)

Density = tf.expand_dims(tf.constant(Density[k]) , 1)
Gravity = tf.expand_dims(tf.constant(Gravity[k]) , 1)


def conv(input, w, strides):
    y = tf.nn.conv1d(input=input, filters=w, stride=strides, padding='SAME')
    # y = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001,center=True,scale=True,
    # beta_initializer='zeros',gamma_initializer='ones',moving_mean_initializer='zeros',moving_variance_initializer='ones' )(y)
    y = tf.nn.leaky_relu(y)
    return y


def deconv(input, w, strides, output):
    y = tf.nn.conv1d_transpose(input=input, filters=w, strides=strides, padding='SAME', output_shape=output)
    # y = tf.keras.layers.BatchNormalization(axis=1,momentum=0.99,epsilon=0.001,center=True,scale=True,
    # beta_initializer='zeros',gamma_initializer='ones',moving_mean_initializer='zeros',moving_variance_initializer='ones' )(y)
    y = tf.nn.leaky_relu(y)
    return y



def fullyConnected_layer(input,w,b):
  y = tf.matmul(input,w) + b
  return y

######################################################################
######################################################################
######################################################################
def Load_WEIGHTS():
    open_file = open('Weights_VAE2.pkl', "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

GenV    =  Load_WEIGHTS()
weights = []
for i in range(9,len(GenV)):
    weights = weights + [ tf.constant(GenV[i].numpy()) ]
Initial = np.random.normal(0, 1, [1,latent_space]).astype('float32')
del(GenV)
######################################################################
######################################################################
######################################################################
def get_tfVariable(name):
    return tf.Variable(Initial, name=name, trainable=True, dtype=tf.float32)


V = []
V = V + [get_tfVariable('W0')]


def decode():
    C = fullyConnected_layer(V[0], weights[0], weights[1])
    C = tf.nn.leaky_relu(C, alpha=1)
    C = fullyConnected_layer(C, weights[2], weights[3])

    C = tf.reshape(C, [C.shape[0], int(np.ceil(N / 4)), 128])
    C = deconv(C, weights[4],1,[C.shape[0],int(np.ceil(N/4)),64])
    C = deconv(C, weights[5], 2, [C.shape[0], int(np.ceil(N / 2)), 32])
    C = deconv(C, weights[6], 2, [C.shape[0], N, 1])
    C = tf.reshape(C,[N,1])
    return C


def loss_function(y_pred):
    Loss = tf.reduce_mean(tf.square(y_pred -Density))
#    Loss = tf.reduce_mean(tf.abs(tf.matmul(A, y_pred) - Gravity))
    return  Loss



def train_step(lr ):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        preds = decode()
        Loss  = loss_function(preds)
        grads = tape.gradient(Loss, V)
        optimizer.apply_gradients(zip(grads, V))
        return Loss

########################################################################################################################
########################################################################################################################
########################################################################################################################
for EPOCH in range(train_epochs):
        avg_Loss = 0
        if np.mod(EPOCH, 2) == 0:
            lr = lr / 2
        for j in range(num_batch):
            Loss = train_step(lr)
            avg_Loss  += (Loss / (num_batch))

        print("--- On DENSITY epoch $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {} ---".format(EPOCH))
        tf.print(" ---Loss:---", avg_Loss)
        print("\n")


B = decode().numpy().reshape([n,n])
C = Density.numpy().reshape([n,n])


# L = encode(Dens[27000:28000])
# sample = latent_sample(L)
# preds = decode(sample)
# preds = np.reshape(preds,[1000,n,n])
# DENS  = np.reshape(Dens[27000:28000] , [1000,n,n])
# def plot():
#
#   for kk in range(100):
#         T = DENS[kk]
#         S = preds[kk]
#
#         DATA1 = np.reshape(T, [n, n])
#         DATA2 = np.reshape(S, [n, n])
#         Z = [DATA1, DATA2]
#         fig, axes = plt.subplots(nrows=2, ncols=1)
#         i = 0
#         for ax in axes.flat:
#             im = ax.imshow(Z[i], extent=[0, 1, 0, 1], vmin=np.min([np.min(Z[0]), np.min(Z[1])]),
#                            vmax=np.max([np.max(Z[0]), np.max(Z[1])]));
#             i = i + 1
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#         fig.colorbar(im, cax=cbar_ax)
#         plt.savefig(str(kk) + '.png')
#         plt.show()
#         plt.close()

#plot()