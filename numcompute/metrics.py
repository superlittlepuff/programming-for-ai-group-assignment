import numpy as np

def accuracy(y_true, y_pred)->float:
    """
    Accuracy = TP+TN/(TP+TN+FP+FN)
    
    Exception:
    1.Input should be array
    2.The number of two arrays should be the same
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_pred == y_true)

def precision(y_true, y_pred)->float:
    """
    Precision = TP/(TP+FP)
    Zero presents Positive
    One presents Negtive
    
    Exception:
    1. TP + FP == 0
    2.Input should be array
    3.The number of two arrays should be the same
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    
    prec = float(TP/(TP + FP))
    return prec

def recall(y_true, y_pred)->float:
    """
    Recall = TP/(TP+FN)
    Zero presents Positive
    One presents Negtive
    
    Exception:
    1. TP + FN == 0
    2.Input should be array
    3.The number of two arrays should be the same
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    
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

    mse_score = np.square(np.subtract(y_true - y_pred)).mean()
    
    return mse_score
    
def roc_curve():
    pass

def auc():
    pass



