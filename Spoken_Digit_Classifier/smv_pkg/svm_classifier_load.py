import torch
from scipy.io import wavfile
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from kymatio import Scattering1D
from os import walk
from joblib import load

###############################################################################
# Pipeline setup
# --------------
# We start by specifying the dimensions of our processing pipeline along with
# some other parameters.
#
# First, we have signal length. Longer signals are truncated and shorter
# signals are zero-padded. The sampling rate is 8000 Hz, so this corresponds to
# little over a second.
T = 2 ** 13

###############################################################################
# Maximum scale 2**J of the scattering transform (here, about 30 milliseconds)
# and the number of wavelets per octave.
J = 8
Q = 12

###############################################################################
# We need a small constant to add to the scattering coefficients before
# computing the logarithm. This prevents very large values when the scattering 
# coefficients are very close to zero.
log_eps = 1e-6

###############################################################################
# If a GPU is available, let's use it!
use_cuda = torch.cuda.is_available()

###############################################################################
# For reproducibility, we fix the seed of the random number generator.
torch.manual_seed(42)

path_dataset = "..\\recordings\\recordings"

files = []
for (dirpath, dirnames, filenames) in walk(path_dataset):
    files.extend(filenames)
    break

###############################################################################
# Set up Tensors to hold the audio signals (`x_all`), the labels (`y_all`), and
# whether the signal is in the train or test set (`subset`).
x_all = torch.zeros(len(files), T, dtype=torch.float32)
y_all = torch.zeros(len(files), dtype=torch.int64)
subset = torch.zeros(len(files), dtype=torch.int64)

###############################################################################
# For each file in the dataset, we extract its label `y` and its index from the
# filename. If the index is between 0 and 4, it is placed in the test set, while
# files with larger indices are used for training. The actual signals are
# normalized to have maximum amplitude one, and are truncated or zero-padded
# to the desired length `T`. They are then stored in the `x_all` Tensor while
# their labels are in `y_all`.
for k, f in enumerate(files):
    basename = f.split('.')[0]

    # Get label (0-9) of recording.
    y = int(basename.split('_')[0])

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

    x_all[k, start:start + x.numel()] = x
    y_all[k] = y

###############################################################################
# Log-scattering transform
# ------------------------
# We now create the `Scattering1D` object that will be used to calculate the
# scattering coefficients.
scattering = Scattering1D(J, T, Q)

###############################################################################
# If we are using CUDA, the scattering transform object must be transferred to
# the GPU by calling its `cuda()` method. The data is similarly transferred.
if use_cuda:
    scattering.cuda()
    x_all = x_all.cuda()
    y_all = y_all.cuda()

###############################################################################
# Compute the scattering transform for all signals in the dataset.
Sx_all = scattering.forward(x_all)

###############################################################################
# Since it does not carry useful information, we remove the zeroth-order
# scattering coefficients, which are always placed in the first channel of
# the scattering Tensor.
Sx_all = Sx_all[:, 1:, :]

###############################################################################
# To increase discriminability, we take the logarithm of the scattering
# coefficients (after adding a small constant to make sure nothing blows up
# when scattering coefficients are close to zero).
Sx_all = torch.log(torch.abs(Sx_all) + log_eps)

###############################################################################
# Finally, we average along the last dimension (time) to get a time-shift
# invariant representation.
Sx_all = torch.mean(Sx_all, dim=-1)

clf = load('models\\poly_3.joblib')

print('\n\nTesting\n\n')

Sx_ts, y_ts = Sx_all, y_all

mu_ts = Sx_ts.mean(dim=0)
std_ts = Sx_ts.std(dim=0)
Sx_ts = (Sx_ts - mu_ts) / std_ts

y_pred = clf.predict(Sx_ts)
score = clf.score(Sx_ts, y_ts)
con_matrix = confusion_matrix(y_ts, y_pred)
cr = classification_report(y_ts, y_pred)
print(score)
print(cr)

plt.figure()
plt.imshow(con_matrix)
tick_locs = np.arange(10)
ticks = ['{}'.format(i) for i in range(1, 11)]
plt.xticks(tick_locs, ticks)
plt.yticks(tick_locs, ticks)
plt.ylabel("True number")
plt.xlabel("Predicted number")
plt.show()
print('1')
