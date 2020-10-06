# CNN: Digit Recognition

The Digit Recognizer is a multi-class classification problem. The data files train.csv and test.csv contain 28*28 pixels gray scale images. We will approach the problem in six major steps:

1. Dataset import and pre-processing
2. Designing CNN architecture
3. Training the networks
4. Predict on test dataset
5. Plotting necessary diagrams
6. Saving output

## PIP Install

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all necessary libraries.

```bash
pip install foobar
```

## DATASET IMPORT AND PRE-PROCESSING

```python
# Libraries
import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import itertools

import seaborn as sns

# Reading dataset
X =  pd.read_csv("../input/train.csv")
X_test_main =  pd.read_csv("../input/test.csv")

# Extract label info
y = X["label"]

X = X.drop(['label'],axis = 1)

# Reshape image matrix
X = X.values.reshape(-1, 28, 28, 1).astype('float32')
X_test_main = X_test_main.values.reshape(-1, 28, 28, 1).astype('float32')

y = y.values
```

```python
# Sample train image plotting:
plt.figure()
plt.imshow( X[1][:,:,0])
plt.colorbar()
plt.grid(False)
plt.show()
```
![](images/__results___5_0.png)
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
