import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


train_epochs    = 25
batch_size      = 25
BIGG_BATCH      = 27000
num_BIGG_BATCH  = 1
num_batch       = int(BIGG_BATCH/batch_size)
n               = 51
N               = n**2
semi            = int(np.ceil(N/4))*128
Z               = 500
latent_space    = 15
lr             = 0.0001



path = Path('/home/cvl/Pycharm/Gravity_Density_Inversion')
Density = np.load(path / ('Density.npy'), allow_pickle=True)
Density = np.expand_dims(Density, axis=2).astype('float32')



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





def get_tfVariable(shape, name, trainable=True):
    return tf.Variable(tf.keras.initializers.GlorotNormal(seed=50,)(shape), name=name, trainable=trainable, dtype=tf.float32)



weights=[]
################################### encode Density
weights = weights + [get_tfVariable([3,1,32],   'W0')]
weights = weights + [get_tfVariable([3,32,64],  'W1')]
weights = weights + [get_tfVariable([3,64,128],  'W3')]


weights = weights + [get_tfVariable([semi,Z],'W5')]
weights = weights + [get_tfVariable([Z],  'W12')]

weights = weights + [get_tfVariable([Z,latent_space],'W5')]
weights = weights + [get_tfVariable([latent_space],  'W12')]
weights = weights + [get_tfVariable([Z,latent_space], 'W6')]
weights = weights + [get_tfVariable([latent_space],  'W13')]

################################### decode Density
weights = weights + [get_tfVariable([latent_space,Z], 'W7')]
weights = weights + [get_tfVariable([Z],    'W11')]

weights = weights + [get_tfVariable([Z,semi], 'W7')]
weights = weights + [get_tfVariable([semi],    'W11')]

weights = weights + [get_tfVariable([3,64,128], 'W9')]
weights = weights + [get_tfVariable([3,32,64], 'W11')]
weights = weights + [get_tfVariable([3,1,32],  'W12')]




def latent_sample(L):
    # L = [mu,var]
    eps     = 1*np.random.normal(0,1,[latent_space])
    sample = L[0] + tf.math.exp(L[1]/2) * eps
    return sample



def encode( C):
     C = conv(C, weights[0], 2)
     C = conv(C, weights[1], 2)
     C = conv(C, weights[2], 1)
     C = tf.reshape(C, [C.shape[0], semi])
     C = fullyConnected_layer(C, weights[3], weights[4])
     C = tf.nn.leaky_relu(C, alpha=1)

     mu = fullyConnected_layer(C, weights[5], weights[6])
     var = fullyConnected_layer(C, weights[7], weights[8])
     return [mu,var]


def decode( C ):
    C = fullyConnected_layer(C, weights[9], weights[10])
    C = tf.nn.leaky_relu(C, alpha=1)
    C = fullyConnected_layer(C, weights[11], weights[12])

    C = tf.reshape(C, [C.shape[0], int(np.ceil(N / 4)), 128])
    C = deconv(C, weights[13],1,[C.shape[0],int(np.ceil(N/4)),64])
    C = deconv(C, weights[14], 2, [C.shape[0], int(np.ceil(N / 2)), 32])
    C = deconv(C, weights[15], 2, [C.shape[0], N, 1])
    return C


def loss_function(L, y_pred, y_true):
    Loss1 = tf.reduce_mean(tf.square(y_pred - y_true))
    Loss2 = tf.reduce_mean(0.5 * tf.reduce_sum( tf.math.exp(L[1]) + L[0]**2 - L[1] ,axis=1))
    Loss  = 1000*Loss1 + 1*Loss2
    return  [Loss1,Loss2,Loss]




def train_step(x_input, lr ):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        L = encode(x_input)
        sample = latent_sample(L)
        preds = decode(sample)
        [Loss1, Loss2, Loss] = loss_function(L, x_input, preds)
        grads = tape.gradient(Loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        return [Loss1,Loss2,Loss]

########################################################################################################################
########################################################################################################################
########################################################################################################################
for EPOCH in range(train_epochs):

        avg_Loss1 = 0 ; avg_Loss2 = 0 ; avg_Loss = 0
        if np.mod(EPOCH, 2) == 0:
            lr = lr / 2

        for j in range(num_batch):
            batch_x = Density[j * batch_size: (j + 1) * batch_size, :, :]
            [Loss1, Loss2, Loss] = train_step(batch_x, lr)
            avg_Loss  += (Loss / (num_batch))
            avg_Loss1 += (Loss1 / (num_batch))
            avg_Loss2 += (Loss2 / (num_batch))

        print("--- On DENSITY epoch $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {} ---".format(EPOCH))
        tf.print(" ---Loss:---", avg_Loss, " ---Loss1:---", avg_Loss1, " ---Loss2:---", avg_Loss2)
        print("\n")

def SAVE_WEIGHTS():
    file_name = "Weights_VAE2.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(weights, open_file)
    open_file.close()


SAVE_WEIGHTS()



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