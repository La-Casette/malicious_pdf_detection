import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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

def kfold_cross_validation(model, X, y, k):
    cv = KFold(n_splits=k, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y.reshape(-1,), scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

def compute_accuracy(y_test, y_test_predicted):
    return metrics.accuracy_score(y_test, y_test_predicted)
