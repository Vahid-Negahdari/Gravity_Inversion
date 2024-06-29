import tensorflow as tf
import numpy as np
from pathlib import Path

path = Path('G:\projet\Gravity Density Inversion')
#########################################################
# Define Hyperparameter
#########################################################
train_epochs = 30
batch_size   = 25
BIGG_BATCH   = 28000
num_batch    = int(BIGG_BATCH/batch_size)
lr           = 0.02
n            = 51
semi         = int(np.ceil(2*n/8)*30)
#########################################################
# Import Data
#########################################################
Density = np.load(path / ('Density.npy'), allow_pickle=True)
Gravity = np.load(path / ('Gravity.npy'), allow_pickle=True)
Gravity = np.expand_dims(Gravity,axis=2)
Gravity = (Gravity-np.min(Gravity))/(np.max(Gravity)-np.min(Gravity))
#########################################################
# Define Some Functions
#########################################################
def conv(input, w, stride):
    y = tf.nn.conv1d(input=input, filters=w, stride=stride,padding='SAME')
    y = tf.nn.leaky_relu(y)
    return y


def fullyConnected_layer(input,w,b):
  y = tf.matmul(input,w) + b
  return y

#########################################################
# Define Weights
#########################################################
def get_tfVariable(shape, name):
    return tf.Variable(tf.keras.initializers.GlorotNormal(seed=14)(shape), name=name, trainable=True, dtype=tf.float32)

weights=[]
weights = weights + [get_tfVariable([3,1,10],   'W0')]
weights = weights + [get_tfVariable([3,10,20],  'W1')]
weights = weights + [get_tfVariable([3,20,30],  'W3')]
#weights = weights + [get_tfVariable([3,128,256],  'W3')]
weights = weights + [get_tfVariable([semi,n**2],'W5')]
weights = weights + [get_tfVariable([n**2],   'W6')]


########################################################
# Define Model
########################################################
def Model(u):
    C = conv(u, weights[0],2)
    C = conv(C, weights[1],2)
    C = conv(C, weights[2],2)
#    C = conv(C, weights[3],2)
    C = tf.reshape(C,[C.shape[0],semi])
    C = fullyConnected_layer(C, weights[3], weights[4])
    return C

#########################################################
# Define Loss Function
#########################################################
def loss_function(y_pred, y_true):
     Loss = tf.reduce_mean(tf.square(y_pred- y_true))
     return  Loss

#######################################################
#######################################################
def train_step(u,uu,lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        preds = Model(u)
        current_loss = loss_function(preds, uu)
        grads = tape.gradient(current_loss, weights )
        optimizer.apply_gradients(zip(grads, weights ))
        return current_loss


def Test_Score(epoch):
    preds = Model(Gravity[27000:28000])
    loss = loss_function(preds, Density[27000:28000])
    print("--- On epoch Test {} ---".format(epoch))
    tf.print(" Loss:",loss)
    print("\n")



################################################################
# Training Process
################################################################
for epoch in range(train_epochs):
      avg_Loss = 0
      if np.mod(epoch,2)==0:
         lr=lr/2

      for s in range(num_batch):
          batch_u   = Density  [s * batch_size  : (s + 1) * batch_size ]
          batch_uu  = Gravity[s * batch_size  : (s + 1) * batch_size ]
          Loss      = train_step(batch_uu, batch_u,lr)
          avg_Loss += Loss / num_batch
      print("--- On epoch {} ---".format(epoch))
      tf.print(" Loss:", avg_Loss)
      print("\n")
      if (epoch % 3 == 0):
         Test_Score(epoch)

