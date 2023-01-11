import sklearn.metrics as metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def compute_scores(y_test, y_test_predicted):
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

def grid_search_kfold_cv(X=None, y=None, model=None, params_grid=None, k=5):
    fnr_lambda = lambda y,y_pred:get_fn(y, y_pred) / (get_fn(y, y_pred) + get_tp(y, y_pred))
    accuracy_lambda = lambda y,y_pred:metrics.accuracy_score(y, y_pred)
    scoring = {"fnr": make_scorer(score_func=fnr_lambda, greater_is_better=False), 
               "accuracy": make_scorer(score_func=accuracy_lambda)}
    estimator = GridSearchCV(estimator=model, param_grid=params_grid, cv=k, scoring=scoring, refit='fnr')
    return estimator.fit(X,y)

def get_fn(y, y_pred):
    return metrics.confusion_matrix(y, y_pred)[1,0]

def get_tp(y, y_pred):
    return metrics.confusion_matrix(y, y_pred)[1,1]
