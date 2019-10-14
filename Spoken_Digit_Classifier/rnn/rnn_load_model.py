
import torch
from torch.nn import NLLLoss
from scipy.io import wavfile
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from kymatio.scattering1d.scattering1d import Scattering1D
from os import walk

# ~ 8KHz sampling rate
T = 2**13

# Wavlets per octet
J = 8
Q = 12

log_eps = 1e-6

use_cuda = torch.cuda.is_available()

torch.manual_seed(42)

path_dataset = "..\\recordings\\recordings\\"
files = []
for (dirpath, dirnames, filenames) in walk(path_dataset):
    files.extend(filenames)
    break

x_all = torch.zeros(len(files), T, dtype=torch.float32)
y_all = torch.zeros(len(files), dtype=torch.int64)
subset = torch.zeros(len(files), dtype=torch.int64)
criterion = NLLLoss()

# Parse Data and Load Into Tensors
for k, f in enumerate(files):
    basename = f.split('.')[0]

    # Get label (0 - 9) of recording.
    y = int(basename.split('_')[0])

    # Note: files are already randomized order
    # Index larger than 5 gets assigned to training set.
    if int(basename.split('_')[2]) >= 5:
        subset[k] = 0
    else:
        subset[k] = 1

    # Load the audio signal and normalize it.
    _, x = wavfile.read(os.path.join(path_dataset, f))
    x = np.asarray(x, dtype='float')
    x /= np.max(np.abs(x))

    # Convert from NumPy array to PyTorch Tensor.
    x = torch.from_numpy(x)

    # If it's too long, truncate it.
    if x.numel() > T:
        x = x[:T]

    # If it's too short, zero-pad it.
    start = (T - x.numel()) // 2

    x_all[k,start:start + x.numel()] = x
    y_all[k] = y

scattering = Scattering1D(J, T, Q)
Sx_all = scattering.forward(x_all)
Sx_all = Sx_all[:,1:,:]  # removes some non-helpful data
Sx_all = torch.log(torch.abs(Sx_all) + log_eps)  # prevents exponential blow up
# average over time (dim -1) to get a time-shift invariant representation
Sx_all = torch.mean(Sx_all, dim=-1)
x = torch.Tensor(np.zeros(Sx_all.shape[-1]))

rnn = torch.load("saved-models/rnn-80.m")
model = torch.load("saved-models/model-80.m")


Sx_te, y_te = Sx_all, y_all
# Use the mean and standard deviation calculated on the training data to standardize the testing data, as well.
mu_te = Sx_te.mean(dim=0)
std_te = Sx_te.std(dim=0)
Sx_te = (Sx_te - mu_te) / std_te

input = Sx_te.view(len(files), 1, -1)
z, hidden = rnn(input)
out = model(z.view(z.size(0), -1))
avg_loss = criterion(out, y_te)

# Try predicting the labels of the signals in the test data and compute the
# accuracy.

y_hat = out.argmax(dim=1)
accu = (y_hat == y_te).float().mean()

print('Test, average loss = {:1.3f}, accuracy = {:1.3f}'.format(avg_loss, accu))

predicted_categories = y_hat.cpu().numpy()
actual_categories = y_te.cpu().numpy()

confusion = confusion_matrix(actual_categories, predicted_categories)
print(classification_report(actual_categories, predicted_categories))
plt.figure()
plt.imshow(confusion)
tick_locs = np.arange(0, 10)
ticks = ['{}'.format(i) for i in range(0, 10)]
plt.xticks(tick_locs, ticks)
plt.yticks(tick_locs, ticks)
plt.ylabel('True Number')
plt.xlabel('Predicted Number')
plt.show()
