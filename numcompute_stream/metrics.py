import numpy as np

def accuracy(y_true, y_pred)->float:
    """
    Compute the accuracy of the predictions.

    Args:
        y_true (array): The true labels.
        y_pred (array): The predictive labels.

    Raises:
        ValueError: if the shape of two arrays are not the same

    Returns:
        float: the accuracy of the predictions.
    """
    
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("The size of two arrays are not the same")
    
    return np.mean(y_pred == y_true)

def precision(y_true, y_pred)->float:
    """
    Compute the precision of the predictions.
    Precision = TP/(TP+FP)

    Args:
        y_true (array): The true labels.
        y_pred (array): The predictive labels.

    Raises:
        ValueError: if the shape of two arrays are not the same

    Returns:
        float: the precision of the predictions.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of two arrays are not the same")
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    
    if TP + FP == 0:
        return 0.0
    
    prec = float(TP/(TP + FP))
    return prec

def recall(y_true, y_pred)->float:
    """
    Compute the recall of the predictions.
    Recall = TP/(TP+FN)

    Args:
        y_true (array): The true labels.
        y_pred (array): The predictive labels.

    Raises:
        ValueError: if the shape of two arrays are not the same

    Returns:
        float: the recall of the predictions.
    """
   
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of two arrays are not the same")
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    
    if TP + FN == 0:
        return 0.0
    
    recall = float(TP/(TP+FN))
    
    return recall

def f1(y_true, y_pred)->float:
    """
    Compute the F1 score of the predictions.
    F1 = 2 * Precision * Recall/(Precision + Recall)

    Args:
        y_true (array): The true labels.
        y_pred (array): The predictive labels.

    Returns:
        float: the F1 score of the predictions.
    """
    prec = precision(y_true=y_true, y_pred=y_pred)
    rec = recall(y_true=y_true, y_pred=y_pred)
    
    if prec + rec == 0:
        return 0.0
    
    f1_score = 2 * (prec*rec) / (prec+rec)
    
    return f1_score
    
def confusion_matrix(y_true, y_pred)->np.array:
    """
    Compute the confusion matrix of the predictions.

    Args:
        y_true (array): The true labels.
        y_pred (array): The predictive labels.

    Raises:
        ValueError: if the shape of two arrays are not the same

    Returns:
        np.array: the confusion matrix of the predictions.
            [[TP, FN],
            [FP, TN]]
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of two arrays are not the same")
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    
    confu_mat = np.array([[TP, FN],[FP, TN]])
    
    return confu_mat

def mse(y_true, y_pred)->float:
    """
    Compute the mean squared error of the predictions.

    Args:
        y_true (array): The true labels.
        y_pred (array): The predictive labels.

    Raises:
        ValueError: if the shape of two arrays are not the same

    Returns:
        float: the mean squared error of the predictions.
    """
    
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of two arrays are not the same")

    mse_score = np.mean(np.square(y_true - y_pred))
    
    return mse_score
    
def roc_curve(y_true, y_score):
    """
    Compute the true positive rate and false positive rate of the predictions based on different thresholds.
    TPR = TP/(TP+FN)
    FPR = FP/(TN+FP)

    Args:
        y_true (array): The true labels.
        y_score (array): The probability scores of the true labels.

    Raises:
        ValueError: if the shape of two arrays are not the same

    Returns:
        np.arrays: the tprs and fprs of the predictions based on different thresholds.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    
    if y_true.shape != y_score.shape:
        raise ValueError("The shape of two arrays are not the same")
    
    thresholds = np.sort(np.unique(y_score))[::-1]
    
    total_pos = (y_true == 1).sum()
    total_neg = (y_true == 0).sum()
    
    tpr = [0.0]
    fpr = [0.0]
    
    for tres in thresholds:
        y_pred = (y_score >= tres).astype(int)
        
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        
        if total_pos > 0:
            tpr.append(TP / total_pos)
        else:
            tpr.append(0.0)
        if total_neg > 0:
            fpr.append(FP / total_neg)
        else:
            fpr.append(0.0)

    return np.array(tpr,dtype=float), np.array(fpr,dtype=float)

def auc(tpr, fpr)->float:
    """
    Compute the area under the curve of the ROC curve.

    Args:
        tpr (array): The true positive rates.
        fpr (array): The false positive rates.

    Returns:
        float: the area under the curve of the ROC curve.
    """
    
    fpr_idx = np.argsort(fpr)
    fpr_sorted = fpr[fpr_idx]
    tpr_sorted = tpr[fpr_idx]
    
    auc_score = np.trapezoid(tpr_sorted, fpr_sorted)
    return auc_score


