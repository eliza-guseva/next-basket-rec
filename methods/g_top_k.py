import pandas as pd
import numpy as np
from metrics import recall_at_k
import preprocess as prp

import logging
logging.basicConfig(level=logging.INFO)


def predict( 
    train: pd.DataFrame,
    test: pd.DataFrame, 
    list_of_k: list[int]
    ) -> pd.DataFrame:
    top_g_100 = train.product_id.value_counts().head(100).index.tolist()
    user_actuals = test.groupby('user_id').agg(
        {'product_id': list,
        'explore_coeff': 'first'
        }
    )
    for k in list_of_k:
        user_actuals.loc[:, f'pred_{k}'] = [top_g_100[:k] for i in range(len(user_actuals))]
    return user_actuals


def eval_recall_at_k(preds, list_of_k):
    evals = preds.copy()
    for k in list_of_k:
        evals.loc[:, f"recall_at_{k}"] = [
            recall_at_k(act_list, pred_list, k) 
            for (act_list, pred_list) 
            in zip(evals.product_id, evals[f'pred_{k}'])
        ]
    return evals[
        ['explore_coeff']
        + [f'recall_at_{k}' for k in list_of_k]
        ]
    
def eval_for_given_sample(sample_ratio: float) -> pd.DataFrame:
    list_of_k = [5, 10, 20]
    train, test = prp.load_train_test(sample_ratio=sample_ratio)
    preds = predict(train, test, list_of_k)
    evals = eval_recall_at_k(preds, list_of_k)
    return evals
    
    
## USAGE ##

def main(): 
    for sample_ratio in [0.1, 0.15, 0.2, 0.25]:
        evals = eval_for_given_sample(sample_ratio)
        print(f"Sample ratio: {sample_ratio}")
        mean = evals.mean(axis=0)
        std = evals.std(axis=0)
        for idx in mean.index:
            print(f"{idx}: {mean[idx]:.3f} +/- {std[idx]:.3f}")
            corr = np.corrcoef(evals.explore_coeff, evals[idx])
            if idx != 'explore_coeff':
                print(f"Correlation with explore_coeff: {corr[0, 1]:.3f}")
        print("-"*100)
        