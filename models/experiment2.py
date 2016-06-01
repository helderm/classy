import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os

epoch = np.linspace(0, 8, 17)

filename = './baseline_correctedDropout/'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err = 1 - val_acc
val1, = plt.plot(epoch, val_err, label='Baseline')

filename = './nooverlap_correctedDropout/'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err = 1 - val_acc
val2, = plt.plot(epoch, val_err, label='NoOverlap')

plt.title('Validation error')
plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.legend(handles=[val1,val2])
plt.show()