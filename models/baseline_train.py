import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os
epoch = np.linspace(0, 8, 17)

plt.subplot(1, 2, 1)
filename = './baseline_correctedDropout/'
val_acc = np.load(os.path.join(filename, 'val_accuracies.npy'))
val_err = 1 - val_acc
val1, = plt.plot(epoch, val_err, label='Baseline')

plt.title('Validation error')
plt.xlabel('Epoch')
plt.ylabel('Error rate')
plt.legend(handles=[val1])


ax1 = plt.subplot(1, 2, 2)
tr_acc = np.load(os.path.join(filename, 'tr_accuracies.npy'))
tr_loss = np.load(os.path.join(filename, 'tr_losses.npy'))
N = 16
avr_loss = np.convolve(tr_loss, np.ones((N,))/N, mode='valid')
loss, = ax1.plot(avr_loss, 'g', label='Training Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax2 = ax1.twinx()
N = 64
avr_acc = np.convolve(tr_acc, np.ones((N,))/N, mode='valid')
print(avr_acc[avr_acc.shape[0]-1])

acc, = ax2.plot(avr_acc, 'b', label='Batch Accuracy')
ax2.set_ylim([0, 1])
ax2.set_ylabel('Accuracy')

plt.legend(handles=[acc, loss])

plt.show()