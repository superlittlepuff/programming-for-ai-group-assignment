import numpy as np

def accuracy(y_true, y_pred)->float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("The size of two arrays are not the same")
    
    return np.mean(y_pred == y_true)

def precision(y_true, y_pred)->float:
    """
    Precision = TP/(TP+FP)
    Zero presents Negative
    One presents Positive
    
    Exception:
    1. TP + FP == 0
    2.Input should be array
    3.The number of two arrays should be the same
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
    Recall = TP/(TP+FN)
    Zero presents Negative
    One presents Positive
    
    Exception:
    1. TP + FN == 0
    2.Input should be array
    3.The number of two arrays should be the same
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
    F1 = 2 * Precision * Recall/(Precision + Recall)
    
    Exception:
    1.Precision + Recall == 0
    2.Input should be array
    3.The number of two arrays should be the same
    """
    prec = precision(y_true=y_true, y_pred=y_pred)
    rec = recall(y_true=y_true, y_pred=y_pred)
    
    if prec + rec == 0:
        return 0.0
    
    f1_score = 2 * (prec*rec) / (prec+rec)
    
    return f1_score
    
def confusion_matrix(y_true, y_pred)->np.array:
    """_summary_

    Args:
        y_true (_array_): _description_
        y_pred (_array_): _description_

    Returns:
        np.array: 
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
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        float: _description_
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of two arrays are not the same")

    mse_score = np.mean(np.square(y_true - y_pred))
    
    return mse_score
    
def roc_curve(y_true, y_score):
    """
    TPR = TP/(TP+FN)
    FPR = FP/(TN+FP)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
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

    return np.array(tpr), np.array(fpr)

def auc(tpr, fpr):
    fpr_idx = np.argsort(fpr)
    fpr_sorted = fpr[fpr_idx]
    tpr_sorted = tpr[fpr_idx]
    
    auc_score = np.trapezoid(tpr_sorted, fpr_sorted)
    return auc_score


