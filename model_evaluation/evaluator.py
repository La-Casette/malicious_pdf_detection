import sklearn.metrics as metrics
# 905: benign benign TN
# 6: benign predicted as malign FP
# 8: malign predicted as benign FN
# 1102: malign malign TP
def compute_scores(y_test, y_test_predicted):
    knn_confmat = metrics.confusion_matrix(y_test, y_test_predicted)
    print('False positive: %d'%(knn_confmat[0, 1]))
    print('True positive: %d'%(knn_confmat[1, 1]))
    print('False negative: %d'%(knn_confmat[1, 0]))
    print('True negative: %d'%(knn_confmat[0, 0]))
    print('False Negative Rate: %f%%'%((knn_confmat[1, 0]/(knn_confmat[1, 0] + knn_confmat[1, 1]))*100))