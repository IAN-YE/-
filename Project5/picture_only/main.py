import numpy as np
from sklearn.model_selection import train_test_split
import torch
import train
import torch.utils.data as Data

if __name__ == '__main__':
    x_train = np.load(file='picture.npy')
    y_train = np.load(file="label.npy")

    x_train = torch.from_numpy(x_train).permute(0,3,1,2).numpy()

    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=123)

    print(X_train.shape)
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_test_t = torch.from_numpy(X_test.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    train_data = Data.TensorDataset(X_train_t, y_train_t)

    test_data = Data.TensorDataset(X_test_t, y_test_t)

    train.run(train_data,test_data)


