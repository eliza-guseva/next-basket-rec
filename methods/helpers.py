import pandas as pd

def get_user_actuals(test: pd.DataFrame):
    return test.groupby('user_id').agg(
        {'product_id': list,
        'explore_coeff': 'first'
        }
    )