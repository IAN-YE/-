import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MLP_relu = pd.read_csv(r'data/MLP1_RELU.csv')
MLP_tanh = pd.read_csv(r'data/MLP1_tanh.csv')
MLP_no = pd.read_csv(r'data/MLP_1_lr.csv')

plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.style.use('seaborn-darkgrid')

plt.plot(MLP_tanh['epoch'], MLP_tanh['train_accuracy'], label="tanh")
plt.plot(MLP_relu['epoch'], MLP_relu['train_accuracy'], label="Relu")
plt.plot(MLP_no['epoch'], MLP_no['train_accuracy'], label="no activation function")

plt.legend(['tanh','ReLU','no'])
plt.title('train_loss with different activation function')
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.show()