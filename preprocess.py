import pandas as pd
import numpy as np
from pathlib import Path
import logging


np.random.seed(1)

logging.basicConfig(level=logging.INFO)


def preprocess(
    dataloc: str = "instacart_dataset",
    sample_ratio: float = 0.1,
    min_count_product: int = 17,
    low_ord_per_user: int = 3,
    high_ord_per_user: int = 50,
) -> None:
    """
    The main function, that more or less follows the preprocessing of
    https://github.com/eliza-guseva/A-Next-Basket-Recommendation-Reality-Check/blob/main/preprocess/Instacart.py
    """
    user_order = ingest_the_data(dataloc)
    sample_data = do_sample_data(user_order, sample_ratio)
    del user_order # it is SO big
    sample_data = remove_outlier_users_by_n_orders(sample_data, low_ord_per_user, high_ord_per_user)
    train, test = get_train_test(sample_data)
    train, test = remove_rare_products(train, test, min_count_product)
    suffix = get_suffix(sample_ratio, min_count_product, low_ord_per_user, high_ord_per_user)
    save_train_test(train, test, dataloc, suffix)
    return train, test


def read_data(fn: str, dataloc: str = "instacart_dataset") -> pd.DataFrame:
    return pd.read_csv(Path(dataloc) / fn)


def ingest_the_data(dataloc: str = "instacart_dataset") -> pd.DataFrame:
    logging.info("Ingesting the data")
    user_order_d = read_data('orders.csv', dataloc)
    order_item_prior = read_data('order_products__prior.csv', dataloc)
    order_item_train = read_data('order_products__train.csv', dataloc)
    order_item = pd.concat([order_item_prior, order_item_train], ignore_index=True)
    user_order = pd.merge(user_order_d, order_item, on='order_id', how='inner')
    logging.info(f"Ingested. Total transactions: {user_order.shape[0]}")
    return user_order


def do_sample_data(user_order: pd.DataFrame, sample_ratio: float = 0.1) -> pd.DataFrame:
    logging.info("Sampling the data")
    user_num = user_order.user_id.nunique()
    user_sample = np.random.choice(user_order.user_id.unique(), int(user_num * sample_ratio), replace=False)
    sample_data = user_order[user_order.user_id.isin(user_sample)].copy()
    logging.info(f"Size of sample data: {sample_data.shape}")
    logging.info(f"Unique users: {sample_data.user_id.nunique()}")
    logging.info(f"Sample ratio: {sample_ratio}")
    return sample_data


def remove_outlier_users_by_n_orders(sample_data: pd.DataFrame, low: int = 3, high: int = 50) -> pd.DataFrame:
    logging.info("Removing outlier users by number of orders")
    user_date = sample_data.groupby('user_id').order_number.apply(lambda x: sorted(set(x)))
    users = user_date[user_date.apply(len).between(low, high)].index
    sd = sample_data[sample_data.user_id.isin(users)].copy()
    logging.info(f"Size of sample data: {sd.shape}")
    logging.info(f"Unique users: {sd.user_id.nunique()}")
    return sd


def _sort_and_drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Sorting and dropping columns")
    df = df.sort_values(['user_id', 'order_number']).reset_index(drop=True)
    if 'explore_coeff' in df.columns:
        df = df[['user_id', 'order_number', 'product_id', 'eval_set', 'explore_coeff']]
    else:
        df = df[['user_id', 'order_number', 'product_id', 'eval_set']]
    return df

def get_train_test(sample_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Getting train and test")
    logging.info(f"Sample data shape: {sample_data.shape}")
    logging.info(f"Unique users: {sample_data.user_id.nunique()}")
    trains = []
    tests = []

    # not all users that have tag 'prior' (meaning train), have also tag 'train' meaning test.
    # we can add this tag manually to user's last order
    user_ids = sample_data.user_id.unique()
    for (idx, user_id) in enumerate(user_ids):
        if idx % 5000 == 0:
            logging.info(f"Processing user {idx} of {len(user_ids)}")
        u_df = sample_data[sample_data.user_id == user_id].sort_values("order_number")
        max_order_number = u_df.order_number.max()
        u_train = u_df[u_df.order_number < max_order_number].copy()
        u_test = u_df[u_df.order_number == max_order_number].copy()
        items_in_test = set(u_test.product_id.unique())
        items_in_train = set(u_train.product_id.unique())
        prop_new_in_test = np.round(len(items_in_test.difference(items_in_train)) / len(items_in_test), 3)
        u_test.loc[:, 'explore_coeff'] = prop_new_in_test

        trains.append(u_train)
        tests.append(u_test)

    train = pd.concat(trains)
    test = pd.concat(tests)
    train['eval_set'] = 'prior'
    test['eval_set'] = 'train'
    
    assert set(test.user_id.unique()).difference(set(train.user_id.unique())) == set()
    assert set(train.user_id.unique()).difference(set(test.user_id.unique())) == set()
    
    logging.info(f"Train shape: {train.shape}")
    logging.info(f"Test shape: {test.shape}")

    return (
        _sort_and_drop_columns(train), 
        _sort_and_drop_columns(test)
    )


def remove_rare_products(
        train: pd.DataFrame, 
        test: pd.DataFrame, 
        min_count_product: int = 17
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove products that appear less than min_count times in the train set.
    """
    logging.info("Removing rare products")
    product_counts = train.product_id.value_counts()
    rare_products = product_counts[product_counts < min_count_product].index
    train = train[~train.product_id.isin(rare_products)].copy()
    test = test[~test.product_id.isin(rare_products)].copy()
    logging.info(f"Removed {len(rare_products)} rare products")
    logging.info(f"Train size: {train.shape}")
    logging.info(f"Test size: {test.shape}")
    return train, test

def get_suffix(
    sample_ratio: float = 0.1,
    min_count_product: int = 17,
    low_ord_per_user: int = 3,
    high_ord_per_user: int = 50,
) -> str:
    sr = str(sample_ratio).replace('.', '')
    return f"sample_{sr}_min_prod_{min_count_product}_low_ord_{low_ord_per_user}_high_ord_{high_ord_per_user}"


def save_train_test(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    dataloc: str = "instacart_dataset",
    suffix: str = ""
) -> None:
    logging.info("Saving train and test")
    train.to_csv(Path(dataloc) / f'train_{suffix}.csv', index=False)
    test.to_csv(Path(dataloc) / f'test_{suffix}.csv', index=False)

