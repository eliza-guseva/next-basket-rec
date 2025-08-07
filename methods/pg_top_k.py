import pandas as pd
import preprocess as prp
from metrics import eval_recall_at_k
import numpy as np



def predict( 
    train: pd.DataFrame,
    test: pd.DataFrame, 
    list_of_k: list[int]
    ) -> pd.DataFrame:
    n = max(list_of_k)
    top_n_for_user = train.groupby('user_id').product_id.apply(lambda x: _find_top_n_for_user(x, n))
    top_n_global = train.product_id.value_counts()[:n].index.tolist()
    user_actuals = test.groupby('user_id').agg(
        {'product_id': list,
        'explore_coeff': 'first'
        }
    )
    for k in list_of_k:
        user_actuals = pd.concat([
            user_actuals,
            top_n_for_user.apply(lambda x: _combine_p_g_top_n(x, top_n_global, k)).rename(f'pred_{k}')
        ], axis=1, join='inner')
    return user_actuals


def eval_for_given_sample(sample_ratio: float) -> pd.DataFrame:
    list_of_k = [5, 10, 20]
    train, test = prp.load_train_test(sample_ratio=sample_ratio)
    preds = predict(train, test, list_of_k)
    evals = eval_recall_at_k(preds, list_of_k)
    return evals


def _find_top_n_for_user(products: list[int], n: int):
    return pd.Series(products).value_counts()[:n].index.tolist()


def _combine_p_g_top_n(top_n_for_user: list[int], top_n_global: list[int], k):
    if len(top_n_for_user) >= k:
        return top_n_for_user[:k]
    else:
        return top_n_for_user + top_n_global[:(k - len(top_n_for_user))]

## USAGE ##

def main(): 
    for sample_ratio in [0.1, 0.25, 0.5, 0.75]:
        evals = eval_for_given_sample(sample_ratio)
        print(f"Sample ratio: {sample_ratio}")
        mean = evals.mean(axis=0)
        std = evals.std(axis=0)
        ci = 1.96 * std / np.sqrt(len(evals))
        for idx in mean.index:
            print(f"{idx}: {mean[idx]:.3f} +/- {ci[idx]:.3f}")
            corr = np.corrcoef(evals.explore_coeff, evals[idx])
            if idx != 'explore_coeff':
                print(f"Correlation with explore_coeff: {corr[0, 1]:.3f}")
        print("-"*100)