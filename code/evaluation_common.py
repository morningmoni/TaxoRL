import codecs

from sklearn import metrics


def output_predictions(predictions_file, relations, predictions, test_set_keys, test_labels, test_scores):
    '''
    Output the model predictions for the test set
    :param predictions_file: the output file path
    :param relations: the ordered list of relations
    :param predictions: the predicted labels for the test set
    :param test_set: the test set - a list of (x, y, sent_x, sent_y, relation, score) instances
    :return:
    '''
    with codecs.open(predictions_file, 'w', 'utf-8') as f_out:
        for i, (x, y) in enumerate(test_set_keys):
            print >> f_out, '\t'.join(
                [x, y, relations[test_labels[i]], relations[predictions[i]], str(test_scores[i][0]),
                 str(test_scores[i][1])])


def evaluate(y_test, y_pred, relations, do_full_reoprt=False):
    '''
    Evaluate performance of the model on the test set
    :param y_test: the test set labels.
    :param y_pred: the predicted values
    :param do_full_reoprt: whether to print the F1, precision and recall of every class.
    :return: mean F1 over all classes
    '''
    if do_full_reoprt:
        full_report(y_test, y_pred, relations)
    pre, rec, f1, support = eval_performance(y_test, y_pred)
    return pre, rec, f1, support


def full_report(y_true, y_pred, relations):
    '''
    Print a full report on the classes performance
    :param y_true: the gold-standard labels
    :param y_pred: the predictions
    :return: the report
    '''
    cr = metrics.classification_report(y_true, y_pred, target_names=relations, digits=3)
    print cr


def eval_performance(y_true, y_pred):
    '''
    Evaluate the performance of a multiclass classification model.
    :param y_true: the gold-standard labels
    :param y_pred: the predictions
    :return: mean F1
    '''
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print '=== Performance ==='
    print 'Mean precision:  %.03f%%' % pre  # (100*sum(pre * support)/sum(support))
    print 'Mean recall:     %.03f%%' % rec  # (100*sum(rec * support)/sum(support))
    print 'Mean F1:         %.03f%%' % f1  # mean_f1
    return pre, rec, f1, support
