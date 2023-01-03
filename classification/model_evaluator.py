import sklearn.metrics as metrics
from sklearn.model_selection import cross_validate

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

def metrics_function(y_true, y_pred):
    fn = (metrics.confusion_matrix(y_true, y_pred)[1,0])/(metrics.confusion_matrix(y_true, y_pred)[1,0] + metrics.confusion_matrix(y_true, y_pred)[1,1])
    f1 = metrics.f1_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return {'false_neg':fn, 'f1_score':f1, 'accuracy_score':accuracy}

def kfold_metrics(clf, X, y):
    y_pred = clf.predict(X)
    return metrics_function(y, y_pred)
    
def kfold_cross_validation(model, X, y, k=5):
    scores = cross_validate(model, X, y, cv=k, scoring=kfold_metrics)
    return scores
