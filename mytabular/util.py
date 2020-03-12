from typing import List, Union, Optional, Callable, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold


SEED = 1019
feature_dir = Path('/home/td009/kaggle-toguro/cfiken/atma4/data/')


def _save_cache(df: pd.DataFrame, name: str, cache_prefix: Optional[str]) -> None:
    if cache_prefix is not None:
        df.to_feather(feature_dir / f'{cache_prefix}_{name}.ftr')


def _use_cache(name: str, cache_prefix: str) -> pd.DataFrame:
    if cache_prefix is not None:
        path = feature_dir / f'{cache_prefix}_{name}.ftr'
        if path.exists():
            feature = pd.read_feather(path)
            return feature


def feature_cache(name: str, is_tuple: bool = False):
    def _cache(func: Callable):
        def wrapper(*args, **kwargs):
            cache_prefix = kwargs.get('cache_prefix')
            do_use_cache = kwargs.get('use_cache', False)
            do_save_cache = kwargs.get('save_cache', False)

            if do_use_cache:
                if is_tuple:
                    train_feature = _use_cache(name, 'train')
                    test_feature = _use_cache(name, 'test')
                    if train_feature is not None and test_feature is not None:
                        train_feature = pd.merge(args[0], train_feature, how='left', on='purchase_id')
                        test_feature = pd.merge(args[1], test_feature, how='left', on='purchase_id')
                        return (train_feature, test_feature), name
                elif cache_prefix is not None:
                    feature = _use_cache(name, cache_prefix)
                    if feature is not None:
                        base = args[0]
                        return pd.merge(base, feature, how='left', on='purchase_id'), name

            feature = func(*args, **kwargs)

            if do_save_cache:
                if is_tuple:
                    _save_cache(feature[0], name, 'train')
                    _save_cache(feature[1], name, 'test')
                elif cache_prefix is not None:
                    _save_cache(feature, name, cache_prefix)

            return feature, name
        return wrapper
    return _cache


def change_column_name(df: Union[pd.DataFrame, pd.Series],
                       old: Union[str, List[str]],
                       new: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(old, str) and isinstance(new, str):
        return df.rename(columns={old: new})

    name_map = {}
    for o, n in zip(old, new):
        name_map[old] = new
    return df.rename(columns=name_map)


def combine_submission_by_is_user_in_train(in_train_submission: pd.DataFrame,
                                           not_in_train_submission: pd.DataFrame,
                                           is_in_train: pd.DataFrame) -> pd.DataFrame:
    '''
    :params in_train_submission: train ユーザが test にもいる前提で作った submission
    :params not_in_train_submission: test にもいる train ユーザに依存しないように (GroupKFold 等) 作った submission
    :params is_in_train: test.csv の並びで、各 purchase_id のユーザが train にもいるかどうかの 0/1 の DataFrame
    :return: 各 purchase_id でユーザが train にもいるかどうかでそれぞれの submission から値をとってきたもの
    '''
    columns = in_train_submission.columns
    for c in columns:
        in_train_submission[c] = in_train_submission[c] * is_in_train['is_in_train']
        not_in_train_submission[c] = not_in_train_submission[c] * (1 - is_in_train['is_in_train'])
    submission = in_train_submission.values + not_in_train_submission.values
    submission = pd.DataFrame(submission, columns=columns)
    return submission


def save_feature_importance(importance: List[np.ndarray],
                            columns: List[str],
                            outputdir: Path,
                            suffix: str):
    df = pd.DataFrame(importance, index=list(columns), columns=['importance'])
    df = df.sort_values('importance', ascending=False)
    df = df.reset_index()
    plt.figure(figsize=(8, 20))
    sns.barplot(x='importance', y='index', data=df)
    plt.tight_layout()
    plt.savefig(outputdir / f'{suffix}_importance.png')


def target_encoding(df_train: pd.DataFrame,
                    target: pd.DataFrame,
                    df_test: pd.DataFrame,
                    columns: str,
                    df_valid: pd.DataFrame = None,
                    kfold: int = 4) -> Tuple[pd.DataFrame, ...]:
    '''
    特徴を一つづつ入れる関数
    カテゴリの特徴量のtarget_encodingを行う。
    kaggle 本p.142
    '''
    # deep copyにしておかないともとの値を置換してしまうことがあるため。
    df_train = df_train.copy().loc[:, columns]
    df_test = df_test.copy().loc[:, columns]
    if df_valid is not None:
        df_valid = df_valid.copy().loc[:, columns]
    # testデータ作成
    data_tmp = pd.DataFrame({columns: df_train, 'target': target})
    target_mean = data_tmp.groupby(columns)["target"].mean()
    df_test = df_test.map(target_mean)
    if df_valid is not None:
        df_valid = df_valid.map(target_mean)
    # trainデータのencoding
    tmp = np.repeat(np.nan, df_train.shape[0])
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    for idx_1, idx_2 in kf.split(df_train):
        target_mean = data_tmp.iloc[idx_1].groupby(columns)[
            'target'].mean()
        tmp[idx_2] = df_train.iloc[idx_2].map(target_mean)
    df_train.loc[:] = tmp
    if df_valid is not None:
        return df_train, df_valid, df_test
    else:
        return df_train, df_test
