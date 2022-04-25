import numpy as np

def l2_norm_diff(func1,func2):
    ldiff = np.sqrt(np.sum((func1-func2)**2))/func1.size
    return ldiff
