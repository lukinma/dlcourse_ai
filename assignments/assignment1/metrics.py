import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    eq_arr = prediction == ground_truth
    n_positives = np.sum(prediction)
    n_correct = np.sum(eq_arr)
    n_true_positives = np.sum((prediction == ground_truth) & (prediction == True))
    n_ground_positives = np.sum(ground_truth)

    precision = n_true_positives / n_positives
    recall = n_true_positives / n_ground_positives
    accuracy = n_correct / len(prediction)
    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    t = prediction == ground_truth
    #print(t, sum(t), sum(t)/len(ground_truth))
    return sum(t)/len(ground_truth)
