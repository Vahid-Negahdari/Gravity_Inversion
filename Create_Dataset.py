import tensorflow as tf
import numpy as np
from pathlib import Path


#path = Path('/home/cvl/Pycharm/Gravity_Density_Inversion')
path = Path('G:\projet\Gravity Density Inversion')
#########################################################
# Set Domain Node
#########################################################
n=51   ; l=20 ; h=(2*l)/(n-1)

dim1 = np.linspace(-l, l, n)
dim2 = np.linspace(-l, l, n)
X,Y  = np.meshgrid(dim1, dim2)
XX   = np.reshape(X,[n*n])
YY   = np.reshape(Y,[n*n])
XXX  = np.zeros([n,n**2])
YYY  = np.zeros([n,n**2])

for i in range(n):
   XXX[i,:] = XX[i]-XX
   YYY[i,:] = YY[i]-10*h-YY


A1 = (h**2)*XXX/((XXX**2 + YYY**2)**(1))
A2 = (h**2)*YYY/((XXX**2 + YYY**2)**(1))
A  = np.concatenate((A1,A2),axis=0).astype('float32')

Density = np.load(path / ('Density_Train.npy' ), allow_pickle=True)
Gravity = tf.transpose(tf.matmul(A,Density.T)).numpy()
np.save(path / ('Gravity.npy'), Gravity)
np.save(path / ('A.npy'), A)
np.save(path / ('Density.npy'), Density)





# for i in range(100):
#     D = np.load(path / ('Density' + str(i) + '.npy' ), allow_pickle=True)
#     G = tf.transpose(tf.matmul(A,D.T)).numpy()
#     np.save(path / ('Gravity' + str(i)), G)
#     np.save(path / ('Density' + str(i)), D)