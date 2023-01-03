import numpy as np
import pandas as pd
def import_data_train_test(dataset = None):
    X_tot = np.load('../../data/%s/X_tot.npy'%(dataset))
    X_train, X_test, y_train, y_test = np.load('../../data/%s/X_train.npy'%(dataset)), np.load('../../data/%s/X_test.npy'%(dataset)), np.load('../../data/%s/y_train.npy'%(dataset)), np.load('../../data/%s/y_test.npy'%(dataset))
    df_tot = pd.read_pickle('../../data/%s/df_tot.pandas'%(dataset))
    return X_tot, df_tot, X_train, X_test, y_train, y_test