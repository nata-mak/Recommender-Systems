import numpy as np

def hit_rate(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)   
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1   
    return hit_rate

def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)[:5] 
    
    hit_rate = (flags.sum() > 0) * 1
    return hit_rate

def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    return precision

def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)    
    recall = flags.sum() / len(bought_list)
    return recall

def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    money_precision = np.dot(flags, prices_recommended) / prices_recommended.sum()
    
    return money_precision

def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    recomended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    bought_list = np.array(bought_list)
    prices_bought = np.array(prices_bought)
    
    flags = np.isin(recommended_list, bought_list)
    
    money_recall = np.dot(flags, prices_recommended) / prices_bought.sum()
    
    return money_recall

def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result
