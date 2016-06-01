import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os


filename = './noconv3_final/test'
epoch = np.linspace(1, 31, 31)/2
acc = np.load(os.path.join(filename, 'test_accuracies.npy'))
err = 1 - acc
err1, = plt.plot(epoch, err, label='Test')

filename = './noconv3_final/train+val'
acc = np.load(os.path.join(filename, 'train_accuracies.npy'))
err = 1 - acc
err2, = plt.plot(epoch, err, label='Training')

plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.legend(handles=[err1,err2])
plt.show()