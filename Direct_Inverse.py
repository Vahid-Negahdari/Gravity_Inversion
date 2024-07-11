import tensorflow as tf
import numpy as np
from pathlib import Path

#path = Path('G:\projet\Gravity Density Inversion')
path = Path('/home/cvl/Pycharm/Gravity_Density_Inversion')
#########################################################
# Define Hyperparameter
#########################################################
train_epochs = 35
batch_size   = 25
BIGG_BATCH   = 27000
num_batch    = int(BIGG_BATCH/batch_size)
lr           = 0.002
n            = 51
semi         = int(np.ceil(1*n/8)*128)
semi2        = int((n**2)/2)
#########################################################
# Import Data
#########################################################
A       = tf.transpose(tf.constant(np.load(path / ('A.npy'), allow_pickle=True)))
Density = np.load(path / ('Density.npy'), allow_pickle=True)
Gravity       = np.load(path / ('Gravity.npy'), allow_pickle=True)
#Gravity = np.zeros([27000,n,2]).astype('float32')
#Gravity[:,:,0] = G[:,0:n]     ;    Gravity[:,:,1] = G[:,n:2*n]
Gravity = np.expand_dims(Gravity,axis=2)
#Gravity = (Gravity-np.min(Gravity))/(np.max(Gravity)-np.min(Gravity))
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
weights = weights + [get_tfVariable([3,1,32],   'W0')]
weights = weights + [get_tfVariable([3,32,64],  'W1')]
weights = weights + [get_tfVariable([3,64,128],  'W3')]
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
def loss_function(y_pred, y_true ,g):
     Loss1 = tf.reduce_mean(tf.abs(y_pred- y_true))
     Loss2 = 0#tf.reduce_mean(tf.square(tf.matmul(y_pred,A)-g[:,:,0] ))
     Loss = Loss1 + Loss2
     return  Loss, Loss1, Loss2

#######################################################
#######################################################
def Update_weights(grads, lr):
    for i in range(5):
        weights[i].assign_sub(lr * grads[i])
#        s, u, v = tf.linalg.svd(weights[i])
#        weights[i].assign(tf.matmul(u, v, transpose_b=True))
        # W_new = -lr*(grads[i] - tf.matmul(tf.matmul(weights[i],grads[i],transpose_a=True,transpose_b=True),weights[i] ))/2
        # weights[i].assign_add(W_new)



def train_step(u,uu,lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    with tf.GradientTape() as tape:
        preds = Model(u)
        L,l1,l2 = loss_function(preds, uu ,u)
        grads = tape.gradient(L, weights )
#        Update_weights(grads,lr)
        optimizer.apply_gradients(zip(grads, weights ))
        return L,l1,l2


def Test_Score(epoch):
    preds = Model(Gravity[27000:28000])
    loss,loss1,loss2 = loss_function(preds, Density[27000:28000], Gravity[27000:28000])
    print("--- On epoch Test {} ---".format(epoch))
    tf.print(" Loss:",loss)
    print("\n")



################################################################
# Training Process
################################################################
for epoch in range(train_epochs):
      avg_Loss = 0
      avg_Loss1 = 0
      avg_Loss2 = 0
      if np.mod(epoch,2)==0:
         lr=lr/2

      for s in range(num_batch):
          batch_u   = Density  [s * batch_size  : (s + 1) * batch_size ]
          batch_uu  = Gravity[s * batch_size  : (s + 1) * batch_size ]
          Loss, Loss1, Loss2      = train_step(batch_uu, batch_u,lr)
          avg_Loss  += Loss  / num_batch
          avg_Loss1 += Loss1 / num_batch
          avg_Loss2 += Loss2 / num_batch
      print("--- On epoch {} ---".format(epoch))
      tf.print(" Loss:", avg_Loss," Loss1:", avg_Loss1, " Loss2:", avg_Loss2)
      print("\n")
      if (epoch % 3 == 0):
         Test_Score(epoch)

