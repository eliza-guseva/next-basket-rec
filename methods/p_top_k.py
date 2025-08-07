import pandas as pd
import preprocess as prp
from metrics import eval_recall_at_k
import numpy as np



def find_top_n(products: list[int], n: int):
    return pd.Series(products).value_counts()[:n].index.tolist()


def predict( 
    train: pd.DataFrame,
    test: pd.DataFrame, 
    list_of_k: list[int]
    ) -> pd.DataFrame:
    n = max(list_of_k)
    top_n_for_user = train.groupby('user_id').product_id.apply(lambda x: find_top_n(x, n))
    user_actuals = test.groupby('user_id').agg(
        {'product_id': list,
        'explore_coeff': 'first'
        }
    )
    for k in list_of_k:
        user_actuals = pd.concat([
            user_actuals,
            top_n_for_user.apply(lambda x: x[:k]).rename(f'pred_{k}')
        ], axis=1, join='inner')
    return user_actuals


def eval_for_given_sample(sample_ratio: float) -> pd.DataFrame:
    list_of_k = [5, 10, 20]
    train, test = prp.load_train_test(sample_ratio=sample_ratio)
    preds = predict(train, test, list_of_k)
    evals = eval_recall_at_k(preds, list_of_k)
    return evals


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