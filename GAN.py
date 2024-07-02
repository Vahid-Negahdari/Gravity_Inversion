import tensorflow as tf
import numpy as np
from pathlib import Path

path = Path('/home/cvl/Pycharm/Elastic_Scattering/Dataset')
#########################################################
# Define Hyperparameter
#########################################################
train_epochs = 45
batch_size   = 25
BIGG_BATCH   = 27000
num_batch    = int(BIGG_BATCH/batch_size)
lr1          = 0.0005
lr2          = 0.0005
n            = 51
k            = 50
semi         = int(np.ceil(n/8)*np.ceil(n/8)*128)
#########################################################
# Import Data
#########################################################
Density = np.load(path / ('Density_Train.npy'), allow_pickle=True).astype('float32')
Density = np.reshape(Density,[27000,n,n,1])
Density = (Density-np.min(Density))/(np.max(Density)-np.min(Density))

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
def get_tfVariable(shape, name):
    return tf.Variable(tf.keras.initializers.GlorotNormal(seed=14)(shape), name=name, trainable=True, dtype=tf.float32)

GenV = []
GenV = GenV + [get_tfVariable([k,int(np.ceil(n/8))*int(np.ceil(n/8))*128]    , 'W5')]
GenV = GenV + [get_tfVariable([int(np.ceil(n/8))*int(np.ceil(n/8))*128]      , 'W6')]
GenV = GenV + [get_tfVariable([3,3,64,128]  , 'W0')]
GenV = GenV + [get_tfVariable([3,3,32,64] , 'W1')]
GenV = GenV + [get_tfVariable([3,3,1,32], 'W3')]



DiscV = []
DiscV = DiscV + [get_tfVariable([3,3,1,32]  , 'W0')]
DiscV = DiscV + [get_tfVariable([3,3,32,64] , 'W1')]
DiscV = DiscV + [get_tfVariable([3,3,64,128], 'W3')]
DiscV = DiscV + [get_tfVariable([semi,1]    , 'W5')]
DiscV = DiscV + [get_tfVariable([1]         , 'W6')]


########################################################
# Define Model
########################################################
def Generator(u):
    C = fullyConnected_layer(u, GenV[0], GenV[1])
    C = tf.reshape(C, [C.shape[0], int(np.ceil(n/8)), int(np.ceil(n/8)), 128])
    C = deconv(C, GenV[2], 2, [C.shape[0], int(np.ceil(n/4)), int(np.ceil(n/4)), 64], 2)
    C = deconv(C, GenV[3], 2, [C.shape[0], int(np.ceil(n/2)), int(np.ceil(n/2)), 32], 2)
    C = deconv(C, GenV[4], 2, [C.shape[0], n, n, 1], 2)
    return C

def Discriminator(u):
    C = conv(u, DiscV[0],2,2)
    C = conv(C, DiscV[1],2,2)
    C = conv(C, DiscV[2],2,2)
    C = tf.reshape(C,[C.shape[0],C.shape[1]*C.shape[2]*C.shape[3]])
    C = fullyConnected_layer(C, DiscV[3], DiscV[4])
    return C

#########################################################
# Define Loss Function
#########################################################
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def Generator_loss(Fake_output):
     return  cross_entropy(tf.ones_like(Fake_output), Fake_output)


def Discriminator_loss(Real_output, Fake_output):
    real_loss = cross_entropy(tf.ones_like(Real_output), Real_output)
    fake_loss = cross_entropy(tf.zeros_like(Fake_output), Fake_output)
    total_loss = real_loss + fake_loss
    return real_loss, fake_loss, total_loss

#######################################################
#######################################################
def train_step(Real,lr1,lr2):
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr1)
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr2)

    Noise = np.random.normal(0,1,[batch_size,k] ).astype('float32')
    with tf.GradientTape() as g_tape , tf.GradientTape() as d_tape:
        Fake        = Generator(Noise)
        Real_output = Discriminator(Real)
        Fake_output = Discriminator(Fake)
        Loss_G      = Generator_loss(Fake_output)
        Loss_Real, Loss_Fake, Loss_D  = Discriminator_loss(Real_output, Fake_output)

    grad_G = g_tape.gradient(Loss_G, GenV )
    grad_D = d_tape.gradient(Loss_D, DiscV)
    optimizer_G.apply_gradients(zip(grad_G, GenV ))
    optimizer_D.apply_gradients(zip(grad_D, DiscV))
    return Loss_G, Loss_D, Loss_Real, Loss_Fake


################################################################
# Training Process
################################################################
for epoch in range(train_epochs):
      avg_Loss1 = 0
      avg_Loss2 = 0
      avg_Loss3 = 0
      avg_Loss4 = 0
      if np.mod(epoch,4)==0:
         lr1=lr1/2 ; lr2=lr2/2

      for s in range(num_batch):
          batch_u         = Density  [s * batch_size  : (s + 1) * batch_size ]
          loss_g, loss_d, loss_real, loss_fake  = train_step(batch_u,lr1,lr2)
          avg_Loss1 += loss_g / num_batch
          avg_Loss2 += loss_d / num_batch
          avg_Loss3 += loss_real / num_batch
          avg_Loss4 += loss_fake / num_batch
      print("--- On epoch {} ---".format(epoch))
      tf.print(" ---Loss1:---", avg_Loss1, " ---Loss2:---", avg_Loss2," ---Loss3:---", avg_Loss3, " ---Loss4:---",avg_Loss4)
      print("\n")
      # if (epoch % 3 == 0):
      #    Test_Score(epoch)

