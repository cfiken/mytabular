# from typing import Optional, Tuple, Callable
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import jpholiday
# import datetime
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction.text import TfidfTransformer

# from mytabular.util import change_column_name, feature_cache, SEED

# '''
# 再現性確保のため特徴作るメソッド切り出してる
# base は index と PurchaseId だけの DataFrame, これに merge するように特徴作成する
# 例:
#     train_xxx_feature = create_xxx(train[['purchase_id']], some_data_frame)
#     test_xxx_feature = create_xxx(test[['purchase_id']], some_data_frame)
# '''

# seed = SEED
# target_columns = [
#     '130123', '130125', '130129', '130131',
#     '140307', '140313', '140316', '140317',
#     '140321', '140501', '140505', '140641', '140691'
# ]


# @feature_cache('user_label')
# def create_user_label(base: pd.DataFrame,
#                       meta: pd.DataFrame,
#                       use_cache: bool = False,
#                       save_cache: bool = False,
#                       cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     user (mpno) を label encode するだけ
#     '''
#     user_le = LabelEncoder()
#     user_le.fit(meta['mpno'])
#     meta['mpno'] = user_le.transform(meta['mpno'])
#     feature = pd.merge(base, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     return feature


# @feature_cache('shop_label')
# def create_shop_label(base: pd.DataFrame,
#                       meta: pd.DataFrame,
#                       use_cache: bool = False,
#                       save_cache: bool = False,
#                       cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     shop (mstr) を label encode するだけ
#     '''
#     shop_le = LabelEncoder()
#     shop_le.fit(meta['mstr'])
#     meta['mstr'] = shop_le.transform(meta['mstr'])
#     feature = pd.merge(base, meta[['purchase_id', 'mstr']], how='left', on='purchase_id')
#     return feature


# @feature_cache('target_in_train')
# def create_target_in_train(base: pd.DataFrame,
#                            meta: pd.DataFrame,
#                            log: pd.DataFrame,
#                            use_cache: bool = False,
#                            save_cache: bool = False,
#                            cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     meta_count = meta.groupby('mpno')['purchase_id'].count()
#     meta_count = change_column_name(meta_count, 'purchase_id', 'p_count')
#     meta = pd.merge(meta, meta_count, how='left', on='mpno')
#     meta_log = pd.merge(meta, log, how='left', on='purchase_id')
#     target_agg = meta_log.groupby(['mpno', 'ccl_category_cd4'])['total'].agg(['count', 'sum'])
#     target_agg['p_count'] = meta_log.groupby(['mpno', 'ccl_category_cd4'])['p_count'].agg('max')
#     target_agg['mean'] = target_agg['count'] / target_agg['p_count']
#     target_agg = target_agg.reset_index()
#     target_agg = target_agg[target_agg['ccl_category_cd4'].isin(target_columns)]
#     target_agg = target_agg.pivot(index='mpno', columns='ccl_category_cd4', values=['sum', 'mean'])
#     target_agg.columns = ['target_' + col[0] + '_' + str(col[1]) for col in target_agg.columns.values]
#     feature = pd.merge(meta[['purchase_id', 'mpno']], target_agg, how='left', on='mpno')
#     feature = feature.drop('mpno', axis=1)
#     feature = feature.fillna(0)
#     feature = pd.merge(base, feature, how='left', on='purchase_id')
#     return feature


# @feature_cache('purchase_together_cd1')
# def create_purchase_together_cd1(base: pd.DataFrame,
#                                  log: pd.DataFrame,
#                                  category: pd.DataFrame,
#                                  use_cache: bool = False,
#                                  save_cache: bool = False,
#                                  cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd1 カテゴリのカウント
#     cd1 カテゴリは3つ (100000, 600000, 700000) だけなので3次元で返し、下記のような DataFrame になる

#     ccl_category_cd1	100000	600000	700000
#     purchase_id
#     222ASAkikq9mATqEn9HwUd	11.0	1.0	1.0
#     222Jy52ZGkfyjSUd8ceVhP	3.0	4.0	0.0
#     222sJuWCq6R664RrEYuWGN	7.0	4.0	0.0
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd1_unique_count = log.groupby('purchase_id')['ccl_category_cd1'].value_counts()
#     cd1_unique_count = change_column_name(cd1_unique_count, 'ccl_category_cd1', 'cd1_together_count')
#     cd1_unique_count = cd1_unique_count.reset_index()
#     cd1_unique_count = cd1_unique_count.pivot(index='purchase_id', columns='ccl_category_cd1', values='cd1_together_count')
#     cd1_unique_count = cd1_unique_count.fillna(0)
#     cd1_unique_count = cd1_unique_count.add_prefix('cd1_count_')
#     feature = pd.merge(base, cd1_unique_count, how='left', on='purchase_id')
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('purchase_together_cd2')
# def create_purchase_together_cd2(base: pd.DataFrame,
#                                  log: pd.DataFrame,
#                                  category: pd.DataFrame,
#                                  use_cache: bool = False,
#                                  save_cache: bool = False,
#                                  cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd2 カテゴリのカウント
#     cd1 と同じだが columns は11個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd2_unique_count = log.groupby('purchase_id')['ccl_category_cd2'].value_counts()
#     cd2_unique_count = change_column_name(cd2_unique_count, 'ccl_category_cd2', 'cd2_together_count')
#     cd2_unique_count = cd2_unique_count.reset_index()
#     cd2_unique_count = cd2_unique_count.pivot(index='purchase_id', columns='ccl_category_cd2', values='cd2_together_count')
#     cd2_unique_count = cd2_unique_count.fillna(0)
#     cd2_unique_count = cd2_unique_count.add_prefix('cd2_count_')
#     feature = pd.merge(base, cd2_unique_count, how='left', on='purchase_id')
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('purchase_together_cd3')
# def create_purchase_together_cd3(base: pd.DataFrame,
#                                  log: pd.DataFrame,
#                                  category: pd.DataFrame,
#                                  use_cache: bool = False,
#                                  save_cache: bool = False,
#                                  cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd3 カテゴリのカウント
#     cd1 と同じだが columns は78個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd3_unique_count = log.groupby('purchase_id')['ccl_category_cd3'].value_counts()
#     cd3_unique_count = change_column_name(cd3_unique_count, 'ccl_category_cd3', 'cd3_together_count')
#     cd3_unique_count = cd3_unique_count.reset_index()
#     cd3_unique_count = cd3_unique_count.pivot(index='purchase_id', columns='ccl_category_cd3', values='cd3_together_count')
#     cd3_unique_count = cd3_unique_count.fillna(0)
#     cd3_unique_count = cd3_unique_count.add_prefix('cd3_count_')
#     feature = pd.merge(base, cd3_unique_count, how='left', on='purchase_id')
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('purchase_together_cd4')
# def create_purchase_together_cd4(base: pd.DataFrame,
#                                  log: pd.DataFrame,
#                                  category: pd.DataFrame,
#                                  use_cache: bool = False,
#                                  save_cache: bool = False,
#                                  cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd4 カテゴリのカウント
#     cd1 と同じだが columns は472個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd4_unique_count = log.groupby('purchase_id')['ccl_category_cd4'].value_counts()
#     cd4_unique_count = change_column_name(cd4_unique_count, 'ccl_category_cd4', 'cd4_together_count')
#     cd4_unique_count = cd4_unique_count.reset_index()
#     cd4_unique_count = cd4_unique_count.pivot(index='purchase_id', columns='ccl_category_cd4', values='cd4_together_count')
#     cd4_unique_count = cd4_unique_count.add_prefix('cd4_count_')
#     cd4_unique_count = cd4_unique_count.fillna(0)
#     feature = pd.merge(base, cd4_unique_count, how='left', on='purchase_id')
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('user_purchase_mean_cd2')
# def create_user_purchase_mean_cd2(base: pd.DataFrame,
#                                   log: pd.DataFrame,
#                                   category: pd.DataFrame,
#                                   meta: pd.DataFrame,
#                                   use_cache: bool = False,
#                                   save_cache: bool = False,
#                                   cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd2 カテゴリのカウントのユーザ平均
#     columns 11個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd2_unique_count = log.groupby('purchase_id')['ccl_category_cd2'].value_counts()
#     cd2_unique_count = change_column_name(cd2_unique_count, 'ccl_category_cd2', 'cd2_user_mean')
#     cd2_unique_count = cd2_unique_count.reset_index()
#     cd2_unique_count = cd2_unique_count.pivot(index='purchase_id', columns='ccl_category_cd2', values='cd2_user_mean')
#     cd2_unique_count = cd2_unique_count.add_prefix('cd2_user_mean_')
#     cd2_unique_count = cd2_unique_count.fillna(0)
#     mpno_count = pd.merge(cd2_unique_count, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     mpno_count = mpno_count.groupby('mpno').agg('mean')
#     feature = pd.merge(base, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     feature = pd.merge(feature, mpno_count, how='left', on='mpno')
#     feature = feature.drop('mpno', axis=1)
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('user_purchase_sum_cd3')
# def create_user_purchase_sum_cd3(base: pd.DataFrame,
#                                  log: pd.DataFrame,
#                                  category: pd.DataFrame,
#                                  meta: pd.DataFrame,
#                                  use_cache: bool = False,
#                                  save_cache: bool = False,
#                                  cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd3 カテゴリのカウントのユーザ合計
#     cd1 と同じだが columns 78個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd3_unique_count = log.groupby('purchase_id')['ccl_category_cd3'].value_counts()
#     cd3_unique_count = change_column_name(cd3_unique_count, 'ccl_category_cd3', 'cd3_together_count')
#     cd3_unique_count = cd3_unique_count.reset_index()
#     cd3_unique_count = cd3_unique_count.pivot(index='purchase_id', columns='ccl_category_cd3', values='cd3_together_count')
#     cd3_unique_count = cd3_unique_count.add_prefix('cd3_user_sum_')
#     cd3_unique_count = cd3_unique_count.fillna(0)
#     mpno_count = pd.merge(cd3_unique_count, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     mpno_count = mpno_count.groupby('mpno').agg('sum')
#     feature = pd.merge(base, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     feature = pd.merge(feature, mpno_count, how='left', on='mpno')
#     feature = feature.drop('mpno', axis=1)
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('user_purchase_mean_cd3')
# def create_user_purchase_mean_cd3(base: pd.DataFrame,
#                                   log: pd.DataFrame,
#                                   category: pd.DataFrame,
#                                   meta: pd.DataFrame,
#                                   use_cache: bool = False,
#                                   save_cache: bool = False,
#                                   cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd3 カテゴリのカウントのユーザ平均
#     columns 78個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd3_unique_count = log.groupby('purchase_id')['ccl_category_cd3'].value_counts()
#     cd3_unique_count = change_column_name(cd3_unique_count, 'ccl_category_cd3', 'cd3_together_count')
#     cd3_unique_count = cd3_unique_count.reset_index()
#     cd3_unique_count = cd3_unique_count.pivot(index='purchase_id', columns='ccl_category_cd3', values='cd3_together_count')
#     cd3_unique_count = cd3_unique_count.add_prefix('cd3_user_mean_')
#     cd3_unique_count = cd3_unique_count.fillna(0)
#     mpno_count = pd.merge(cd3_unique_count, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     mpno_count = mpno_count.groupby('mpno').agg('mean')
#     feature = pd.merge(base, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     feature = pd.merge(feature, mpno_count, how='left', on='mpno')
#     feature = feature.drop('mpno', axis=1)
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('user_purchase_sum_cd4')
# def create_user_purchase_sum_cd4(base: pd.DataFrame,
#                                  log: pd.DataFrame,
#                                  category: pd.DataFrame,
#                                  meta: pd.DataFrame,
#                                  use_cache: bool = False,
#                                  save_cache: bool = False,
#                                  cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd4 カテゴリのカウントのユーザ合計
#     cd1 と同じだが columns は472個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd4_unique_count = log.groupby('purchase_id')['ccl_category_cd4'].value_counts()
#     cd4_unique_count = change_column_name(cd4_unique_count, 'ccl_category_cd4', 'cd4_together_count')
#     cd4_unique_count = cd4_unique_count.reset_index()
#     cd4_unique_count = cd4_unique_count.pivot(index='purchase_id', columns='ccl_category_cd4', values='cd4_together_count')
#     cd4_unique_count = cd4_unique_count.add_prefix('cd4_user_sum_')
#     cd4_unique_count = cd4_unique_count.fillna(0)
#     mpno_count = pd.merge(cd4_unique_count, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     mpno_count = mpno_count.groupby('mpno').agg('sum')
#     feature = pd.merge(base, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     feature = pd.merge(feature, mpno_count, how='left', on='mpno')
#     feature = feature.drop('mpno', axis=1)
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('user_purchase_mean_cd4')
# def create_user_purchase_mean_cd4(base: pd.DataFrame,
#                                   log: pd.DataFrame,
#                                   category: pd.DataFrame,
#                                   meta: pd.DataFrame,
#                                   use_cache: bool = False,
#                                   save_cache: bool = False,
#                                   cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id について、 target 以外で同時に購入したものの cd4 カテゴリのカウントのユーザ平均
#     cd1 と同じだが columns は472個ある
#     '''
#     log = pd.merge(log, category, how='left', on='ccl_category_cd4')
#     # target は抜く
#     log = log[~log['ccl_category_cd4'].isin(target_columns)]

#     cd4_unique_count = log.groupby('purchase_id')['ccl_category_cd4'].value_counts()
#     cd4_unique_count = change_column_name(cd4_unique_count, 'ccl_category_cd4', 'cd4_together_count')
#     cd4_unique_count = cd4_unique_count.reset_index()
#     cd4_unique_count = cd4_unique_count.pivot(index='purchase_id', columns='ccl_category_cd4', values='cd4_together_count')
#     cd4_unique_count = cd4_unique_count.add_prefix('cd4_user_mean_')
#     cd4_unique_count = cd4_unique_count.fillna(0)
#     mpno_count = pd.merge(cd4_unique_count, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     mpno_count = mpno_count.groupby('mpno').agg('mean')
#     feature = pd.merge(base, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     feature = pd.merge(feature, mpno_count, how='left', on='mpno')
#     feature = feature.drop('mpno', axis=1)
#     feature = feature.fillna(0)
#     return feature


# @feature_cache('purchase_together_cd3_pca4', is_tuple=True)
# def create_purchase_together_cd3_pca4(base_train: pd.DataFrame,
#                                       base_test: pd.DataFrame,
#                                       log: pd.DataFrame,
#                                       category: pd.DataFrame,
#                                       use_cache: bool = False,
#                                       save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_purchase_cd3(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase_cd3', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd3_tfidf_pca4', is_tuple=True)
# def create_purchase_together_cd3_tfidf_pca4(base_train: pd.DataFrame,
#                                             base_test: pd.DataFrame,
#                                             log: pd.DataFrame,
#                                             category: pd.DataFrame,
#                                             use_cache: bool = False,
#                                             save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_purchase_cd3(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase_cd3', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd3_pca8', is_tuple=True)
# def create_purchase_together_cd3_pca8(base_train: pd.DataFrame,
#                                       base_test: pd.DataFrame,
#                                       log: pd.DataFrame,
#                                       category: pd.DataFrame,
#                                       use_cache: bool = False,
#                                       save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_purchase_cd3(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase_cd3', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd3_tfidf_pca8', is_tuple=True)
# def create_purchase_together_cd3_tfidf_pca8(base_train: pd.DataFrame,
#                                             base_test: pd.DataFrame,
#                                             log: pd.DataFrame,
#                                             category: pd.DataFrame,
#                                             use_cache: bool = False,
#                                             save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_purchase_cd3(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase_cd3', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd3_pca16', is_tuple=True)
# def create_purchase_together_cd3_pca16(base_train: pd.DataFrame,
#                                        base_test: pd.DataFrame,
#                                        log: pd.DataFrame,
#                                        category: pd.DataFrame,
#                                        use_cache: bool = False,
#                                        save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_purchase_cd3(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase_cd3', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd3_tfidf_pca16', is_tuple=True)
# def create_purchase_together_cd3_tfidf_pca16(base_train: pd.DataFrame,
#                                              base_test: pd.DataFrame,
#                                              log: pd.DataFrame,
#                                              category: pd.DataFrame,
#                                              use_cache: bool = False,
#                                              save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_purchase_cd3(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase_cd3', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_pca4', is_tuple=True)
# def create_purchase_together_cd4_pca4(base_train: pd.DataFrame,
#                                       base_test: pd.DataFrame,
#                                       log: pd.DataFrame,
#                                       category: pd.DataFrame,
#                                       use_cache: bool = False,
#                                       save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_pca8', is_tuple=True)
# def create_purchase_together_cd4_pca8(base_train: pd.DataFrame,
#                                       base_test: pd.DataFrame,
#                                       log: pd.DataFrame,
#                                       category: pd.DataFrame,
#                                       use_cache: bool = False,
#                                       save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 8 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_pca16', is_tuple=True)
# def create_purchase_together_cd4_pca16(base_train: pd.DataFrame,
#                                        base_test: pd.DataFrame,
#                                        log: pd.DataFrame,
#                                        category: pd.DataFrame,
#                                        use_cache: bool = False,
#                                        save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 16 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_pca32', is_tuple=True)
# def create_purchase_together_cd4_pca32(base_train: pd.DataFrame,
#                                        base_test: pd.DataFrame,
#                                        log: pd.DataFrame,
#                                        category: pd.DataFrame,
#                                        use_cache: bool = False,
#                                        save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を PCA で 32 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 32
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_tfidf_pca4', is_tuple=True)
# def create_purchase_together_cd4_tfidf_pca4(base_train: pd.DataFrame,
#                                             base_test: pd.DataFrame,
#                                             log: pd.DataFrame,
#                                             category: pd.DataFrame,
#                                             use_cache: bool = False,
#                                             save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を tfidf 変換し PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_tfidf_pca8', is_tuple=True)
# def create_purchase_together_cd4_tfidf_pca8(base_train: pd.DataFrame,
#                                             base_test: pd.DataFrame,
#                                             log: pd.DataFrame,
#                                             category: pd.DataFrame,
#                                             use_cache: bool = False,
#                                             save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を tfidf 変換し PCA で 8 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_tfidf_pca16', is_tuple=True)
# def create_purchase_together_cd4_tfidf_pca16(base_train: pd.DataFrame,
#                                              base_test: pd.DataFrame,
#                                              log: pd.DataFrame,
#                                              category: pd.DataFrame,
#                                              use_cache: bool = False,
#                                              save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を tfidf 変換し PCA で 16 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_together_cd4_tfidf_pca32', is_tuple=True)
# def create_purchase_together_cd4_tfidf_pca32(base_train: pd.DataFrame,
#                                              base_test: pd.DataFrame,
#                                              log: pd.DataFrame,
#                                              category: pd.DataFrame,
#                                              use_cache: bool = False,
#                                              save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を tfidf 変換し PCA で 32 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 32
#     train_feature, test_feature = _do_pca_purchase_cd4(
#         n_components, base_train, base_test, log, category,
#         column_name='purchase', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_sum_cd4_pca4', is_tuple=True)
# def create_user_purchase_sum_cd4_pca4(base_train: pd.DataFrame,
#                                       base_test: pd.DataFrame,
#                                       log: pd.DataFrame,
#                                       category: pd.DataFrame,
#                                       meta: pd.DataFrame,
#                                       use_cache: bool = False,
#                                       save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で aggregate し PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='sum', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_sum_cd4_pca8', is_tuple=True)
# def create_user_purchase_sum_cd4_pca8(base_train: pd.DataFrame,
#                                       base_test: pd.DataFrame,
#                                       log: pd.DataFrame,
#                                       category: pd.DataFrame,
#                                       meta: pd.DataFrame,
#                                       use_cache: bool = False,
#                                       save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で aggregate し PCA で 8 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='sum', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_sum_cd4_pca16', is_tuple=True)
# def create_user_purchase_sum_cd4_pca16(base_train: pd.DataFrame,
#                                        base_test: pd.DataFrame,
#                                        log: pd.DataFrame,
#                                        category: pd.DataFrame,
#                                        meta: pd.DataFrame,
#                                        use_cache: bool = False,
#                                        save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で aggregate し PCA で 16 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='sum', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_sum_cd4_tfidf_pca4', is_tuple=True)
# def create_user_purchase_sum_cd4_tfidf_pca4(base_train: pd.DataFrame,
#                                             base_test: pd.DataFrame,
#                                             log: pd.DataFrame,
#                                             category: pd.DataFrame,
#                                             meta: pd.DataFrame,
#                                             use_cache: bool = False,
#                                             save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で aggregate し PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='sum', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_sum_cd4_tfidf_pca8', is_tuple=True)
# def create_user_purchase_sum_cd4_tfidf_pca8(base_train: pd.DataFrame,
#                                             base_test: pd.DataFrame,
#                                             log: pd.DataFrame,
#                                             category: pd.DataFrame,
#                                             meta: pd.DataFrame,
#                                             use_cache: bool = False,
#                                             save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で aggregate し PCA で 8 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='sum', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_sum_cd4_tfidf_pca16', is_tuple=True)
# def create_user_purchase_sum_cd4_tfidf_pca16(base_train: pd.DataFrame,
#                                              base_test: pd.DataFrame,
#                                              log: pd.DataFrame,
#                                              category: pd.DataFrame,
#                                              meta: pd.DataFrame,
#                                              use_cache: bool = False,
#                                              save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で aggregate し PCA で 16 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='sum', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_mean_cd4_pca4', is_tuple=True)
# def create_user_purchase_mean_cd4_pca4(base_train: pd.DataFrame,
#                                        base_test: pd.DataFrame,
#                                        log: pd.DataFrame,
#                                        category: pd.DataFrame,
#                                        meta: pd.DataFrame,
#                                        use_cache: bool = False,
#                                        save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で mean aggregate し PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='mean', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_mean_cd4_pca8', is_tuple=True)
# def create_user_purchase_mean_cd4_pca8(base_train: pd.DataFrame,
#                                        base_test: pd.DataFrame,
#                                        log: pd.DataFrame,
#                                        category: pd.DataFrame,
#                                        meta: pd.DataFrame,
#                                        use_cache: bool = False,
#                                        save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で mean aggregate し PCA で 8 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='mean', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_mean_cd4_pca16', is_tuple=True)
# def create_user_purchase_mean_cd4_pca16(base_train: pd.DataFrame,
#                                         base_test: pd.DataFrame,
#                                         log: pd.DataFrame,
#                                         category: pd.DataFrame,
#                                         meta: pd.DataFrame,
#                                         use_cache: bool = False,
#                                         save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で mean aggregate し PCA で 16 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='mean', use_tfidf=False, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_mean_cd4_tfidf_pca4', is_tuple=True)
# def create_user_purchase_mean_cd4_tfidf_pca4(base_train: pd.DataFrame,
#                                              base_test: pd.DataFrame,
#                                              log: pd.DataFrame,
#                                              category: pd.DataFrame,
#                                              meta: pd.DataFrame,
#                                              use_cache: bool = False,
#                                              save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で mean aggregate し
#     TFIDF を PCA で 4 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 4
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='mean', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_mean_cd4_tfidf_pca8', is_tuple=True)
# def create_user_purchase_mean_cd4_tfidf_pca8(base_train: pd.DataFrame,
#                                              base_test: pd.DataFrame,
#                                              log: pd.DataFrame,
#                                              category: pd.DataFrame,
#                                              meta: pd.DataFrame,
#                                              use_cache: bool = False,
#                                              save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で mean aggregate し
#     TFIDF を PCA で 8 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 8
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='mean', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('user_purchase_mean_cd4_tfidf_pca16', is_tuple=True)
# def create_user_purchase_mean_cd4_tfidf_pca16(base_train: pd.DataFrame,
#                                               base_test: pd.DataFrame,
#                                               log: pd.DataFrame,
#                                               category: pd.DataFrame,
#                                               meta: pd.DataFrame,
#                                               use_cache: bool = False,
#                                               save_cache: bool = False) -> pd.DataFrame:
#     '''
#     create_purchase_together_cd4 で作成した特徴を user で mean aggregate し
#     TFIDF を PCA で 16 に次元削減
#     input に余計なカラムが入っていない base を使う必要がある
#     '''
#     assert base_train.shape[1] == 1, f'base_train size is not 1, {base_train.shape[1]}'
#     assert base_test.shape[1] == 1, f'base_test size is not 1, {base_test.shape[1]}'
#     n_components = 16
#     train_feature, test_feature = _do_pca_user_cd4(
#         n_components, base_train, base_test, log, category, meta,
#         column_name='user_purchase', agg_name='mean', use_tfidf=True, use_cache=use_cache)
#     return train_feature, test_feature


# @feature_cache('purchase_date')
# def create_purchase_date(base: pd.DataFrame,
#                          log: pd.DataFrame,
#                          meta: pd.DataFrame,
#                          use_cache: bool = False,
#                          save_cache: bool = False,
#                          cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id の日付を取得する
#     '''
#     log = pd.merge(log, meta, how='left', on='purchase_id')
#     log = log.groupby('purchase_id')['p_date'].max()
#     feature = pd.merge(base, log, how='left', on='purchase_id')
#     return feature


# @feature_cache('weekday')
# def create_weekday(base: pd.DataFrame,
#                    meta: pd.DataFrame,
#                    use_cache: bool = False,
#                    save_cache: bool = False,
#                    cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id の日付を取得する
#     '''
#     meta['p_date'] = pd.to_datetime(meta['p_date'])
#     meta['weekday'] = meta['p_date'].apply(lambda x: x.weekday())
#     feature = pd.merge(base, meta[['purchase_id', 'weekday']], how='left', on='purchase_id')
#     return feature


# @feature_cache('date_holiday')
# def create_date_info(base: pd.DataFrame,
#                      meta: pd.DataFrame,
#                      use_cache: bool = False,
#                      save_cache: bool = False,
#                      cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id の日付が休日かどうか、及び休前日かどうかを取得する。
#     '''
#     def is_holiday(x):
#         weekend = x.weekday() >= 5
#         holiday = jpholiday.is_holiday(x)
#         yearend = x.month == 12 and x.day in [29, 30, 31]
#         return weekend or holiday or yearend

#     def is_holiday_eve(x):
#         tomorrow = x + datetime.timedelta(days=1)
#         is_today_holiday = is_holiday(x)
#         is_tomorrow_holiday = is_holiday(tomorrow)
#         return not is_today_holiday and is_tomorrow_holiday

#     meta['p_date'] = pd.to_datetime(meta['p_date'])
#     meta['is_holiday'] = meta['p_date'].apply(is_holiday).astype(int)
#     meta['is_holiday_eve'] = meta['p_date'].apply(is_holiday_eve).astype(int)
#     meta = meta[['purchase_id', 'is_holiday', 'is_holiday_eve']]
#     feature = pd.merge(base, meta, how='left', on='purchase_id')
#     return feature


# @feature_cache('purchase_time')
# def create_purchase_time(base: pd.DataFrame,
#                          log: pd.DataFrame,
#                          meta: pd.DataFrame,
#                          use_cache: bool = False,
#                          save_cache: bool = False,
#                          cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id の時間を取得する
#     '''
#     log = pd.merge(log, meta, how='left', on='purchase_id')
#     log = log.groupby('purchase_id')['p_time'].max()
#     feature = pd.merge(base, log, how='left', on='purchase_id')
#     return feature


# @feature_cache('time_ap15')
# def create_time_ap15(base: pd.DataFrame,
#                      meta: pd.DataFrame,
#                      use_cache: bool = False,
#                      save_cache: bool = False,
#                      cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id の時間が 15~3時かどうかのフラグを取得する
#     '''
#     meta['time_ap_15'] = np.logical_or(meta['p_time'].values >= 15,
#                                        meta['p_time'].values <= 3).astype(int)
#     feature = pd.merge(base,
#                        meta[['purchase_id', 'time_ap_15']],
#                        how='left',
#                        on='purchase_id')
#     return feature


# @feature_cache('last_date_diff')
# def create_last_date_diff(base: pd.DataFrame,
#                           meta: pd.DataFrame,
#                           use_cache: bool = False,
#                           save_cache: bool = False,
#                           cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id のユーザの、前回の購買から今回の購買までの日数を取得する
#     '''
#     meta['p_date'] = pd.to_datetime(meta['p_date'])
#     meta['last_date_diff'] = meta.groupby('mpno')['p_date'].diff(1).dt.days
#     meta = meta[['purchase_id', 'last_date_diff']]
#     feature = pd.merge(base, meta, how='left', on='purchase_id')
#     return feature


# @feature_cache('user_1week_purchase_count')
# def create_1week_purchase_count(base: pd.DataFrame,
#                                 meta: pd.DataFrame,
#                                 use_cache: bool = False,
#                                 save_cache: bool = False,
#                                 cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id のユーザについて、過去1週間の購買数を取得する。
#     ターゲットの購買情報は含まれていない。
#     '''
#     date_df = pd.DataFrame(pd.date_range('2017-01-02', '2018-12-31', freq='D').date, columns=['p_date'])
#     date_df['p_date'] = pd.to_datetime(date_df['p_date'])
#     mpno_df = pd.DataFrame(meta['mpno'].unique(), columns=['mpno'])
#     date_df['key'] = 0
#     mpno_df['key'] = 0
#     date_mpno_df = date_df.merge(mpno_df, on='key').drop('key', axis=1)

#     date_mpno_count = meta.groupby(['p_date', 'mpno'])['purchase_id'].count().reset_index()
#     date_mpno_count['p_date'] = pd.to_datetime(date_mpno_count['p_date'])
#     date_mpno_count = change_column_name(date_mpno_count, 'purchase_id', 'purchase_count')

#     count_df = date_mpno_df.merge(date_mpno_count, how='left', on=['p_date', 'mpno'])
#     count_df = count_df.fillna(0)
#     count_df['purchase_cumsum'] = count_df.groupby('mpno')['purchase_count'].cumsum()
#     count_df['1week_ago_cumsum'] = count_df.groupby('mpno')['purchase_cumsum'].shift(7)
#     count_df['1week_purchase_count'] = count_df['purchase_cumsum'] - count_df['1week_ago_cumsum'] - 1
#     meta['p_date'] = pd.to_datetime(meta['p_date'])
#     feature = pd.merge(meta, count_df, how='left', on=['p_date', 'mpno'])
#     feature = feature[['purchase_id', '1week_purchase_count']]
#     feature = pd.merge(base, feature, how='left', on='purchase_id')
#     return feature


# @feature_cache('user_4weeks_purchase_count')
# def create_4week_purchase_count(base: pd.DataFrame,
#                                 meta: pd.DataFrame,
#                                 use_cache: bool = False,
#                                 save_cache: bool = False,
#                                 cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id のユーザについて、過去4週間の購買数を取得する。
#     ターゲットの購買情報は含まれていない。
#     '''
#     date_df = pd.DataFrame(pd.date_range('2017-01-02', '2018-12-31', freq='D').date, columns=['p_date'])
#     date_df['p_date'] = pd.to_datetime(date_df['p_date'])
#     mpno_df = pd.DataFrame(meta['mpno'].unique(), columns=['mpno'])
#     date_df['key'] = 0
#     mpno_df['key'] = 0
#     date_mpno_df = date_df.merge(mpno_df, on='key').drop('key', axis=1)

#     date_mpno_count = meta.groupby(['p_date', 'mpno'])['purchase_id'].count().reset_index()
#     date_mpno_count['p_date'] = pd.to_datetime(date_mpno_count['p_date'])
#     date_mpno_count = change_column_name(date_mpno_count, 'purchase_id', 'purchase_count')

#     count_df = date_mpno_df.merge(date_mpno_count, how='left', on=['p_date', 'mpno'])
#     count_df = count_df.fillna(0)
#     count_df['purchase_cumsum'] = count_df.groupby('mpno')['purchase_count'].cumsum()
#     count_df['4week_ago_cumsum'] = count_df.groupby('mpno')['purchase_cumsum'].shift(28)
#     count_df['4week_purchase_count'] = count_df['purchase_cumsum'] - count_df['4week_ago_cumsum'] - 1
#     meta['p_date'] = pd.to_datetime(meta['p_date'])
#     feature = pd.merge(meta, count_df, how='left', on=['p_date', 'mpno'])
#     feature = feature[['purchase_id', '4week_purchase_count']]
#     feature = pd.merge(base, feature, how='left', on='purchase_id')
#     return feature


# @feature_cache('user_cd3_7days_mean')
# def create_user_cd3_7days_mean(base: pd.DataFrame,
#                                meta: pd.DataFrame,
#                                use_cache: bool = False,
#                                save_cache: bool = False,
#                                cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id のユーザについて、過去4週間の購買数を取得する。
#     ターゲットの購買情報は含まれていない。
#     '''
#     return


# @feature_cache('user_cd3_28days_mean')
# def create_user_cd3_28days_mean(base: pd.DataFrame,
#                                 meta: pd.DataFrame,
#                                 use_cache: bool = False,
#                                 save_cache: bool = False,
#                                 cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     ある purchase_id のユーザについて、過去4週間の購買数を取得する。
#     ターゲットの購買情報は含まれていない。
#     '''
#     return


# @feature_cache('is_in_train')
# def create_is_in_train(base_train: pd.DataFrame,
#                        base_test: pd.DataFrame,
#                        meta: pd.DataFrame,
#                        use_cache: bool = False,
#                        save_cache: bool = False,
#                        cache_prefix: Optional[str] = None) -> pd.DataFrame:
#     '''
#     meta の purchase_id について、train にいるユーザによるものかそうでないものかを返す。
#     他の特徴作成メソッドと違い、返り値は meta がベースの test 期間のみの DataFrame になっているので注意。
#     '''
#     base_train['is_in_train'] = 1
#     meta_in_train = pd.merge(meta, base_train, how='left', on='purchase_id')
#     meta_in_train = meta_in_train.fillna(0)
#     # まだ train の期間のユーザしか入ってないので、user でまとめる
#     user_in_train = meta_in_train.groupby('mpno')['is_in_train'].max().to_frame()
#     # meta に戻す
#     meta_user = pd.merge(meta, user_in_train, how='left', on='mpno')
#     # meta_user['p_date'] = pd.to_datetime(meta_user['p_date'])
#     # test_meta = meta_user.query('p_date >= "2018-04-01"')
#     # test_meta = test_meta[['purchase_id', 'is_in_train']]
#     test = pd.merge(base_test, meta_user[['purchase_id', 'is_in_train']], how='left', on='purchase_id')
#     return test


# def _do_pca(n_components: int,
#             features: pd.DataFrame,
#             train_feature: pd.DataFrame,
#             test_feature: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     '''
#     train, test の特徴から PCA を適用して次元削減した特徴を返す
#     '''
#     pca = PCA(n_components=n_components, random_state=seed)
#     pca.fit(features)
#     train_feature = pca.transform(train_feature)
#     test_feature = pca.transform(test_feature)
#     return train_feature, test_feature


# def _do_pca_purchase_cd3(n_components: int,
#                          base_train: pd.DataFrame,
#                          base_test: pd.DataFrame,
#                          log: pd.DataFrame,
#                          category: pd.DataFrame,
#                          column_name: str,
#                          use_tfidf: bool,
#                          use_cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     '''
#     cd4 のカウント特徴について PCA を適用して次元削減した特徴を返す
#     '''
#     train_feature, _ = create_purchase_together_cd3(
#         base_train, log, category, use_cache=use_cache, cache_prefix='train')
#     train_feature = train_feature.set_index('purchase_id')
#     test_feature, _ = create_purchase_together_cd3(
#         base_test, log, category, use_cache=use_cache, cache_prefix='test')
#     test_feature = test_feature.set_index('purchase_id')
#     all_feature = pd.concat([train_feature, test_feature], axis=0)
#     if use_tfidf:
#         tfidf = TfidfTransformer()
#         tfidf.fit(all_feature)
#         all_feature = tfidf.transform(all_feature).toarray()
#         train_feature = tfidf.transform(train_feature).toarray()
#         test_feature = tfidf.transform(test_feature).toarray()
#         column_name = f'tfidf_{column_name}'
#     train_feature, test_feature = _do_pca(n_components, all_feature, train_feature, test_feature)
#     columns = [f'{column_name}_pca_{i}' for i in range(n_components)]
#     base_train[columns] = pd.DataFrame(train_feature, columns=columns)
#     base_test[columns] = pd.DataFrame(test_feature, columns=columns)
#     return base_train, base_test


# def _do_pca_purchase_cd4(n_components: int,
#                          base_train: pd.DataFrame,
#                          base_test: pd.DataFrame,
#                          log: pd.DataFrame,
#                          category: pd.DataFrame,
#                          column_name: str,
#                          use_tfidf: bool,
#                          use_cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     '''
#     cd4 のカウント特徴について PCA を適用して次元削減した特徴を返す
#     '''
#     train_feature, _ = create_purchase_together_cd4(
#         base_train, log, category, use_cache=use_cache, cache_prefix='train')
#     train_feature = train_feature.set_index('purchase_id')
#     test_feature, _ = create_purchase_together_cd4(
#         base_test, log, category, use_cache=use_cache, cache_prefix='test')
#     test_feature = test_feature.set_index('purchase_id')
#     all_feature = pd.concat([train_feature, test_feature], axis=0)
#     if use_tfidf:
#         tfidf = TfidfTransformer()
#         tfidf.fit(all_feature)
#         all_feature = tfidf.transform(all_feature).toarray()
#         train_feature = tfidf.transform(train_feature).toarray()
#         test_feature = tfidf.transform(test_feature).toarray()
#         column_name = f'tfidf_{column_name}'
#     train_feature, test_feature = _do_pca(n_components, all_feature, train_feature, test_feature)
#     columns = [f'{column_name}_pca_{i}' for i in range(n_components)]
#     base_train[columns] = pd.DataFrame(train_feature, columns=columns)
#     base_test[columns] = pd.DataFrame(test_feature, columns=columns)
#     return base_train, base_test


# def _do_pca_user_cd4(n_components: int,
#                      base_train: pd.DataFrame,
#                      base_test: pd.DataFrame,
#                      log: pd.DataFrame,
#                      category: pd.DataFrame,
#                      meta: pd.DataFrame,
#                      column_name: str,
#                      agg_name: str,
#                      use_tfidf: bool,
#                      use_cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     '''
#     cd4 のユーザごとのカウント特徴について PCA を適用して次元削減した特徴を返す。
#     通常の特徴では purchase_id が index になっているため複数ユーザが含まれるので、
#     ユーザを index にして PCA を適用するため、create_purchase_together_cd4 から作り直している
#     '''
#     train_feature, _ = create_purchase_together_cd4(
#         base_train, log, category, use_cache=use_cache, cache_prefix='train')
#     train_feature = train_feature.set_index('purchase_id')
#     test_feature, _ = create_purchase_together_cd4(
#         base_test, log, category, use_cache=use_cache, cache_prefix='test')
#     test_feature = test_feature.set_index('purchase_id')
#     all_features = pd.concat([train_feature, test_feature], axis=0)
#     all_mpno = pd.merge(all_features, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     train_mpno = pd.merge(train_feature, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')
#     test_mpno = pd.merge(test_feature, meta[['purchase_id', 'mpno']], how='left', on='purchase_id')

#     all_features = all_mpno.groupby('mpno').agg(agg_name)  # purchase_id は消える
#     train_feature = pd.merge(train_mpno[['purchase_id', 'mpno']], all_features, how='left', on='mpno')
#     test_feature = pd.merge(test_mpno[['purchase_id', 'mpno']], all_features, how='left', on='mpno')
#     train_index = train_feature[['purchase_id', 'mpno']]
#     test_index = test_feature[['purchase_id', 'mpno']]
#     train_feature = train_feature.drop(['purchase_id', 'mpno'], axis=1)
#     test_feature = test_feature.drop(['purchase_id', 'mpno'], axis=1)
#     column_name = column_name + '_' + agg_name

#     if use_tfidf:
#         tfidf = TfidfTransformer()
#         tfidf.fit(all_features)
#         all_features = tfidf.transform(all_features).toarray()
#         train_feature = tfidf.transform(train_feature).toarray()
#         test_feature = tfidf.transform(test_feature).toarray()
#         column_name = f'tfidf_{column_name}'

#     train_feature, test_feature = _do_pca(n_components, all_features, train_feature, test_feature)
#     columns = [f'{column_name}_pca_{i}' for i in range(n_components)]
#     train_feature = pd.DataFrame(train_feature, columns=columns)
#     train_feature = pd.concat([train_index, train_feature], axis=1).drop('mpno', axis=1)
#     test_feature = pd.DataFrame(test_feature, columns=columns)
#     test_feature = pd.concat([test_index, test_feature], axis=1).drop('mpno', axis=1)
#     return train_feature, test_feature


# if __name__ == '__main__':
#     train = pd.read_csv('/home/td009/kaggle-toguro/data/train.csv')
#     test = pd.read_csv('/home/td009/kaggle-toguro/data/test.csv')
#     meta = pd.read_csv('/home/td009/kaggle-toguro/data/meta.csv')
#     log = pd.read_csv('/home/td009/kaggle-toguro/data/purchase_log.csv')
#     category = pd.read_csv('/home/td009/kaggle-toguro/data/category.csv')

#     base_train = train.copy()[['purchase_id']]
#     base_test = test.copy()[['purchase_id']]

#     print(create_purchase_date(base_train, log, meta))
