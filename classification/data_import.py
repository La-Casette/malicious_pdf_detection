import numpy as np
import pandas as pd
def import_data_train_test():
    X_tot = np.load('../data/X_tot.npy')
    X_train, X_test, y_train, y_test = np.load('../data/X_train.npy'), np.load('../data/X_test.npy'), np.load('../data/y_train.npy'), np.load('../data/y_test.npy')
    df_tot = pd.read_pickle('../data/df_tot.pandas')
    return X_tot, df_tot, X_train, X_test, y_train, y_test