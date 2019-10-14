'''
pytorch classification of spoken digits
'''
import torch
from torch.nn import Linear, NLLLoss, LogSoftmax, Sequential
from torch.nn import RNN
from torch.optim import Adam
from scipy.io import wavfile
import os
import numpy as np
from sklearn.metrics import confusion_matrix
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

path_dataset = "..\\recordings\\"
files = []
for (dirpath, dirnames, filenames) in walk(path_dataset):
    files.extend(filenames)
    break

x_all = torch.zeros(len(files), T, dtype=torch.float32)
y_all = torch.zeros(len(files), dtype=torch.int64)
subset = torch.zeros(len(files), dtype=torch.int64)


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
#x = x + 10
#Sx_all = torch.cat((Sx_all, x), dim=0)

# Select Training Data.
Sx_tr, y_tr = Sx_all[subset == 0], y_all[subset == 0]

# Set Mean to 0, and variance to 1. -> Normal Distribution
mu_tr = Sx_tr.mean(dim=0)
std_tr = Sx_tr.std(dim=0)
Sx_tr = (Sx_tr - mu_tr) / std_tr

# Design ML Model
num_inputs = Sx_tr.shape[-1]
num_classes = y_tr.cpu().unique().numel()
model = Sequential(Linear(num_inputs, num_classes), LogSoftmax(dim=1))
rnn = RNN(336, 336)
optimizer = Adam(model.parameters())
criterion = NLLLoss()

if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# Number of signals to use in each gradient descent step (batch).
batch_size = 32
# Number of epochs.
num_epochs = 80
# Learning rate for Adam.
lr = 1e-2

# set number of batches
nsamples = Sx_tr.shape[0]
nbatches = nsamples // batch_size

hist = []
for e in range(num_epochs):
    # Randomly permute the data. If necessary, transfer the permutation to the
    # GPU.
    perm = torch.randperm(nsamples)
    if use_cuda:
        perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take
    # one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (1+i) * batch_size]
        model.zero_grad()
        rnn.zero_grad()
        input = Sx_tr[idx].view(Sx_tr[idx].size(0), 1, -1)
        # input size. 32 x 1 x 336
        z, hidden = rnn(input)
        out = model(z.view(z.size(0), -1))  # z... torch.Size([32, 336])
        # out size. 32 x 10
        loss = criterion(out, y_tr[idx])
        loss.backward()
        optimizer.step()

    # Calculate the response of the training data at the end of this epoch and
    # the average loss.
    input = Sx_tr.view(1800, 1, -1)
    z, hidden = rnn(input)
    out = model(z.view(z.size(0), -1))
    avg_loss = criterion(out, y_tr)
    hist.append(avg_loss)

    # Try predicting the classes of the signals in the training set and compute
    # the accuracy.
    y_hat = out.argmax(dim=1)
    accuracy = (y_tr == y_hat).float().mean()
    print('Epoch {}, average loss = {:1.3f}, accuracy = {:1.3f}'.format(e, avg_loss, accuracy))
# TESTING
#  we extract the test data (those for which subset equals 1) and the associated labels.
Sx_te, y_te = Sx_all[subset == 1], y_all[subset == 1]
# Use the mean and standard deviation calculated on the training data to standardize the testing data, as well.
Sx_te = (Sx_te - mu_tr) / std_tr

input = Sx_te.view(200, 1, -1)
z, hidden = rnn(input)
out = model(z.view(z.size(0), -1))
avg_loss = criterion(out, y_te)

torch.save(rnn, "saved-models/rnn-70.m")
torch.save(model, "saved-models/model-70.m")

# Try predicting the labels of the signals in the test data and compute the
# accuracy.

y_hat = out.argmax(dim=1)
accu = (y_hat == y_te).float().mean()

print('Test, average loss = {:1.3f}, accuracy = {:1.3f}'.format(avg_loss, accu))

predicted_categories = y_hat.cpu().numpy()
actual_categories = y_te.cpu().numpy()

confusion = confusion_matrix(actual_categories, predicted_categories)
plt.figure()
plt.imshow(confusion)
tick_locs = np.arange(10)
ticks = ['{}'.format(i) for i in range(1, 11)]
plt.xticks(tick_locs, ticks)
plt.yticks(tick_locs, ticks)
plt.ylabel('True Number')
plt.xlabel('Predicted Number')
plt.show()

plt.clf()
plt.title('loss')
plt.plot(hist)
plt.show()


