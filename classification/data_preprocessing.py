import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
class KMeansDimensionalityReduction():
    def __init__(self, df):
        # Separate benign and malware
        df_ben = df[df['malware'] == False]
        df_ben = df_ben.drop(['malware'], axis=1)
        df_mal = df[df['malware'] == True]
        df_mal = df_mal.drop(['malware'], axis=1)
        # Convert each value > 0 into 1
        for col in df_ben.columns:
            df_ben.loc[df_ben[col] > 0, col] = 1
            df_mal.loc[df_mal[col] > 0, col] = 1
        # Sum over the rows
        self.f_ben = np.sum(df_ben.to_numpy(), axis=0).reshape(-1,1) # f_ben[0] = # of benign pdf with that feature (at least one inside)
        self.f_mal = np.sum(df_mal.to_numpy(), axis=0).reshape(-1,1)
        # Apply kmeans to get most relevant features separately for benign and malign
        kmeans_ben = KMeans(n_clusters=2, random_state=0).fit(self.f_ben)
        highest_cluster_center = np.argmax(kmeans_ben.cluster_centers_)
        self.f_ben_rel = kmeans_ben.labels_ == highest_cluster_center
        kmeans_mal = KMeans(n_clusters=2, random_state=0).fit(self.f_mal)
        highest_cluster_center = np.argmax(kmeans_mal.cluster_centers_)
        self.f_mal_rel = kmeans_mal.labels_ == highest_cluster_center
        self.rel_mask_df = np.hstack((self.f_ben_rel | self.f_mal_rel,[True]))
        self.rel_mask = self.f_ben_rel | self.f_mal_rel

    def fit_ben_mal_kmeans(self, df=None, X_tot=None, X_train=None):
        return df.iloc[:,self.rel_mask_df], X_tot[:,self.rel_mask], X_train[:,self.rel_mask]
    
    def transform_ben_mal_kmeans(self, X_test):
        return X_test[:,self.rel_mask]

    def get_kmeans_features(self):
        return self.f_ben_rel, self.f_mal_rel