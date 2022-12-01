import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.model_selection import train_test_split
def compute_scores(y_test, y_test_predicted):
    """
        905: benign benign TN
        6: benign predicted as malign FP
        8: malign predicted as benign FN
        1102: malign malign TP
    """
    knn_confmat = metrics.confusion_matrix(y_test, y_test_predicted)
    print('False positive: %d'%(knn_confmat[0, 1]))
    print('True positive: %d'%(knn_confmat[1, 1]))
    print('False negative: %d'%(knn_confmat[1, 0]))
    print('True negative: %d'%(knn_confmat[0, 0]))
    print('False Negative Rate: %f%%'%((knn_confmat[1, 0]/(knn_confmat[1, 0] + knn_confmat[1, 1]))*100))

def false_negative_loss_function(clf, X, y):
    y_pred = clf.predict(X)
    fn = metrics.confusion_matrix(y, y_pred)[1,0]
    f1 = metrics.f1_score(y, y_pred)
    return {'false_neg':fn, 'f1_score':f1}
    
def kfold_cross_validation(model, X, y, k=5):
    scores = cross_validate(model, X, y, cv=k, scoring=false_negative_loss_function)
    #cv = KFold(n_splits=k, random_state=1, shuffle=True)
    #if scoring == 'precision':
    #    scores = cross_val_score(model, X, y.reshape(-1,), 
    #                            scoring=metrics.make_scorer(metrics.precision_score, pos_label=0), # NPV metric
    #                            cv=cv, 
    #                            n_jobs=-1)
    return scores

def compute_accuracy(y_test, y_test_predicted):
    return metrics.accuracy_score(y_test, y_test_predicted)

def compute_f1_score(y_test, y_test_predicted):
    return metrics.f1_score(y_test, y_test_predicted)

def compute_fpr(y_test, y_test_predicted):
    return metrics.precision_score(y_test, y_test_predicted, pos_label=0)

def get_train_test_split(df):
    X_tot = df.iloc[:,:-1]
    y_tot = df.iloc[:,-1:]
    X_train, X_test, y_train, y_test= train_test_split(X_tot, y_tot,
                                                        test_size= 0.2,
                                                        shuffle= True, #shuffle the data to avoid bias
                                                        stratify=df['malware'],
                                                        random_state= 0)
    return X_train, X_test, y_train, y_test

