# import os
# from typing import Tuple
# import pandas as pd
# import numpy as np
# import pytest

# from mytabular.feature.feature import (
#     create_user_label,
#     create_shop_label,
#     create_target_in_train,
#     create_purchase_together_cd1,
#     create_purchase_together_cd2,
#     create_purchase_together_cd3,
#     create_purchase_together_cd4,
#     create_user_purchase_sum_cd3,
#     create_user_purchase_sum_cd4,
#     create_user_purchase_mean_cd2,
#     create_user_purchase_mean_cd3,
#     create_user_purchase_mean_cd4,
#     create_purchase_together_cd3_pca4,
#     create_purchase_together_cd3_pca8,
#     create_purchase_together_cd3_pca16,
#     create_purchase_together_cd3_tfidf_pca4,
#     create_purchase_together_cd3_tfidf_pca8,
#     create_purchase_together_cd3_tfidf_pca16,
#     create_purchase_together_cd4_pca4,
#     create_purchase_together_cd4_pca8,
#     create_purchase_together_cd4_pca16,
#     create_purchase_together_cd4_tfidf_pca4,
#     create_purchase_together_cd4_tfidf_pca8,
#     create_purchase_together_cd4_tfidf_pca16,
#     create_user_purchase_sum_cd4_pca4,
#     create_user_purchase_sum_cd4_pca8,
#     create_user_purchase_sum_cd4_pca16,
#     create_user_purchase_sum_cd4_tfidf_pca4,
#     create_user_purchase_sum_cd4_tfidf_pca8,
#     create_user_purchase_sum_cd4_tfidf_pca16,
#     create_user_purchase_mean_cd4_pca4,
#     create_user_purchase_mean_cd4_pca8,
#     create_user_purchase_mean_cd4_pca16,
#     create_user_purchase_mean_cd4_tfidf_pca4,
#     create_user_purchase_mean_cd4_tfidf_pca8,
#     create_user_purchase_mean_cd4_tfidf_pca16,
#     create_purchase_date,
#     create_weekday,
#     create_date_info,
#     create_purchase_time,
#     create_time_ap15,
#     create_last_date_diff,
#     create_1week_purchase_count,
#     create_4week_purchase_count,
#     create_user_cd3_7days_mean,
#     create_user_cd3_28days_mean
# )

# # control test
# test_all = False
# use_cache = True
# save_cache = False

# is_ci = 'CI' in os.environ and os.environ['CI'] == 'true'
# do_not_test = is_ci or not test_all

# # if not is_ci:
# #     train = pd.read_csv('/home/td009/kaggle-toguro/data/train.csv')
# #     test = pd.read_csv('/home/td009/kaggle-toguro/data/test.csv')
# #     meta_base = pd.read_csv('/home/td009/kaggle-toguro/data/meta.csv')
# #     log_base = pd.read_csv('/home/td009/kaggle-toguro/data/purchase_log.csv')
# #     category_base = pd.read_csv('/home/td009/kaggle-toguro/data/category.csv')


# def get_base() -> Tuple:
#     return train.copy()[['purchase_id']], test.copy()[['purchase_id']]


# def _base_shape_test(base, feature, num_feature: int = 1):
#     assert np.all(base['purchase_id'].values == feature['purchase_id'].values)
#     assert feature.shape[0] == base.shape[0]
#     assert feature.shape[1] == num_feature + 1


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_label():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_label(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_label(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     assert len(train_feature['mpno'].unique()) == 2543
#     assert len(test_feature['mpno'].unique()) == 4509


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_shop_label():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_shop_label(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_shop_label(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     assert len(train_feature['mstr'].unique()) == 7
#     assert len(test_feature['mstr'].unique()) == 7


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_target_in_train():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     log = log_base.copy()
#     train_feature, _ = create_target_in_train(
#         base_train, meta, log, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_target_in_train(
#         base_test, meta, log, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 26)
#     _base_shape_test(base_test, test_feature, 26)

#     # contents
#     train_c = train_feature.query('purchase_id == "222ASAkikq9mATqEn9HwUd"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['target_sum_130123'].values[0] == 18
#     assert train_c['target_mean_130131'].values[0] == 0.2777777777777778
#     assert test_c['target_sum_130123'].values[0] == 3
#     assert test_c['target_mean_130131'].values[0] == 0.011111111111111112


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd1():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     train_feature, _ = create_purchase_together_cd1(
#         base_train, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_purchase_together_cd1(
#         base_test, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 3)
#     _base_shape_test(base_test, test_feature, 3)

#     # contents
#     train_c = train_feature.query('purchase_id == "222ASAkikq9mATqEn9HwUd"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd1_count_100000'].values[0] == 11.0
#     assert train_c['cd1_count_600000'].values[0] == 1.0
#     assert train_c['cd1_count_700000'].values[0] == 1.0
#     assert test_c['cd1_count_100000'].values[0] == 7
#     assert test_c['cd1_count_600000'].values[0] == 9
#     assert test_c['cd1_count_700000'].values[0] == 2


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd2():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     train_feature, _ = create_purchase_together_cd2(
#         base_train, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_purchase_together_cd2(
#         base_test, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 11)
#     _base_shape_test(base_test, test_feature, 11)

#     # contents
#     train_c = train_feature.query('purchase_id == "222ASAkikq9mATqEn9HwUd"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd2_count_110000'].values[0] == 9.0
#     assert train_c['cd2_count_610000'].values[0] == 1.0
#     assert train_c['cd2_count_710000'].values[0] == 0.0
#     assert test_c['cd2_count_110000'].values[0] == 7
#     assert test_c['cd2_count_610000'].values[0] == 8
#     assert test_c['cd2_count_710000'].values[0] == 0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd3():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     train_feature, _ = create_purchase_together_cd3(
#         base_train, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_purchase_together_cd3(
#         base_test, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 78)
#     _base_shape_test(base_test, test_feature, 78)

#     # contents
#     train_c = train_feature.query('purchase_id == "222ASAkikq9mATqEn9HwUd"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd3_count_110100'].values[0] == 1.0
#     assert train_c['cd3_count_720300'].values[0] == 1.0
#     assert test_c['cd3_count_110100'].values[0] == 1
#     assert test_c['cd3_count_720300'].values[0] == 0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd3_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd3_pca4(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['purchase_cd3_pca_0'].values[0], float)
#     assert isinstance(test_feature['purchase_cd3_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd3_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd3_pca8(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['purchase_cd3_pca_0'].values[0], float)
#     assert isinstance(test_feature['purchase_cd3_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd3_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd3_pca16(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['purchase_cd3_pca_0'].values[0], float)
#     assert isinstance(test_feature['purchase_cd3_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd3_tfidf_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd3_tfidf_pca4(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_purchase_cd3_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_purchase_cd3_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd3_tfidf_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd3_tfidf_pca8(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_purchase_cd3_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_purchase_cd3_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd3_tfidf_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd3_tfidf_pca16(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_purchase_cd3_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_purchase_cd3_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_purchase_cd3_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd4():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     train_feature, _ = create_purchase_together_cd4(
#         base_train, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_purchase_together_cd4(
#         base_test, log, category, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 472)
#     _base_shape_test(base_test, test_feature, 472)

#     # contents
#     train_c = train_feature.query('purchase_id == "229wMfkvnu98jMRhe5PucL"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd4_count_110101'].values[0] == 1.0
#     assert train_c['cd4_count_730205'].values[0] == 0.0
#     assert test_c['cd4_count_110101'].values[0] == 1.0
#     assert test_c['cd4_count_730205'].values[0] == 0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd3():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_purchase_sum_cd3(
#         base_train, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_purchase_sum_cd3(
#         base_test, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 78)
#     _base_shape_test(base_test, test_feature, 78)

#     # contents
#     train_c = train_feature.query('purchase_id == "229wMfkvnu98jMRhe5PucL"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd3_user_sum_110100'].values[0] == 7.0
#     assert train_c['cd3_user_sum_730200'].values[0] == 0.0
#     assert test_c['cd3_user_sum_110100'].values[0] == 51.0
#     assert test_c['cd3_user_sum_730200'].values[0] == 29.0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd4():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_purchase_sum_cd4(
#         base_train, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_purchase_sum_cd4(
#         base_test, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 472)
#     _base_shape_test(base_test, test_feature, 472)

#     # contents
#     train_c = train_feature.query('purchase_id == "229wMfkvnu98jMRhe5PucL"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd4_user_sum_110101'].values[0] == 1.0
#     assert train_c['cd4_user_sum_730205'].values[0] == 0.0
#     assert test_c['cd4_user_sum_110101'].values[0] == 4.0
#     assert test_c['cd4_user_sum_730205'].values[0] == 0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd2():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_purchase_mean_cd2(
#         base_train, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_purchase_mean_cd2(
#         base_test, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 11)
#     _base_shape_test(base_test, test_feature, 11)

#     # contents
#     train_c = train_feature.query('purchase_id == "229wMfkvnu98jMRhe5PucL"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd2_user_mean_110000'].values[0] == 2.4375
#     assert train_c['cd2_user_mean_730000'].values[0] == 0.0625
#     assert test_c['cd2_user_mean_110000'].values[0] == 3.8222222222222224
#     assert test_c['cd2_user_mean_730000'].values[0] == 0.5111111111111111


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd3():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_purchase_mean_cd3(
#         base_train, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_purchase_mean_cd3(
#         base_test, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 78)
#     _base_shape_test(base_test, test_feature, 78)

#     # contents
#     train_c = train_feature.query('purchase_id == "229wMfkvnu98jMRhe5PucL"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd3_user_mean_110100'].values[0] == 0.4375
#     assert train_c['cd3_user_mean_730200'].values[0] == 0.0
#     assert test_c['cd3_user_mean_110100'].values[0] == 0.5666666666666667
#     assert test_c['cd3_user_mean_730200'].values[0] == 0.32222222222222224


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd4():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_purchase_mean_cd4(
#         base_train, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_purchase_mean_cd4(
#         base_test, log, category, meta,
#         use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 472)
#     _base_shape_test(base_test, test_feature, 472)

#     # contents
#     train_c = train_feature.query('purchase_id == "229wMfkvnu98jMRhe5PucL"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['cd4_user_mean_110101'].values[0] == 0.0625
#     assert train_c['cd4_user_mean_730205'].values[0] == 0.0
#     assert test_c['cd4_user_mean_110101'].values[0] == 0.044444444444444446
#     assert test_c['cd4_user_mean_730205'].values[0] == 0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd4_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd4_pca4(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'purchase_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'purchase_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['purchase_pca_0'].values[0], float)
#     assert isinstance(test_feature['purchase_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd4_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd4_pca8(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'purchase_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'purchase_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['purchase_pca_0'].values[0], float)
#     assert isinstance(test_feature['purchase_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd4_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd4_pca16(
#         base_train, base_test, log, category,
#         use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'purchase_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'purchase_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['purchase_pca_0'].values[0], float)
#     assert isinstance(test_feature['purchase_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd4_tfidf_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd4_tfidf_pca4(
#         base_train, base_test, log, category, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_purchase_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_purchase_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_purchase_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_purchase_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd4_tfidf_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd4_tfidf_pca8(
#         base_train, base_test, log, category, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_purchase_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_purchase_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_purchase_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_purchase_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_together_cd4_tfidf_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     features, _ = create_purchase_together_cd4_tfidf_pca16(
#         base_train, base_test, log, category, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_purchase_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_purchase_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_purchase_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_purchase_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd4_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_sum_cd4_pca4(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['user_purchase_sum_pca_0'].values[0], float)
#     assert isinstance(test_feature['user_purchase_sum_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd4_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_sum_cd4_pca8(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['user_purchase_sum_pca_0'].values[0], float)
#     assert isinstance(test_feature['user_purchase_sum_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd4_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_sum_cd4_pca16(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['user_purchase_sum_pca_0'].values[0], float)
#     assert isinstance(test_feature['user_purchase_sum_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd4_tfidf_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_sum_cd4_tfidf_pca4(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_user_purchase_sum_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_user_purchase_sum_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd4_tfidf_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_sum_cd4_tfidf_pca8(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_user_purchase_sum_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_user_purchase_sum_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_sum_cd4_tfidf_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_sum_cd4_tfidf_pca16(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_user_purchase_sum_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_user_purchase_sum_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_user_purchase_sum_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd4_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_mean_cd4_pca4(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['user_purchase_mean_pca_0'].values[0], float)
#     assert isinstance(test_feature['user_purchase_mean_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd4_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_mean_cd4_pca8(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['user_purchase_mean_pca_0'].values[0], float)
#     assert isinstance(test_feature['user_purchase_mean_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd4_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_mean_cd4_pca16(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['user_purchase_mean_pca_0'].values[0], float)
#     assert isinstance(test_feature['user_purchase_mean_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd4_tfidf_pca4():
#     n_components = 4
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_mean_cd4_tfidf_pca4(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_user_purchase_mean_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_user_purchase_mean_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd4_tfidf_pca8():
#     n_components = 8
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_mean_cd4_tfidf_pca8(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_user_purchase_mean_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_user_purchase_mean_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_purchase_mean_cd4_tfidf_pca16():
#     n_components = 16
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     category = category_base.copy()
#     meta = meta_base.copy()
#     features, _ = create_user_purchase_mean_cd4_tfidf_pca16(
#         base_train, base_test, log, category, meta, use_cache=use_cache, save_cache=save_cache
#     )
#     train_feature, test_feature = features

#     # basic shape test
#     _base_shape_test(base_train, train_feature, n_components)
#     _base_shape_test(base_test, test_feature, n_components)

#     # contents
#     assert train_feature.columns[1:].tolist() == [f'tfidf_user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert test_feature.columns[1:].tolist() == [f'tfidf_user_purchase_mean_pca_{i}' for i in range(n_components)]
#     assert isinstance(train_feature['tfidf_user_purchase_mean_pca_0'].values[0], float)
#     assert isinstance(test_feature['tfidf_user_purchase_mean_pca_0'].values[0], float)


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_date():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_purchase_date(
#         base_train, log, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_purchase_date(
#         base_test, log, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     train_c = train_feature.query('purchase_id == "njibeyLPrsnu4HCopjBihW"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['p_date'].values[0] == '2017-01-02'
#     assert test_c['p_date'].values[0] == '2018-04-01'


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_weekday():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_weekday(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_weekday(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     train_c = train_feature.query('purchase_id == "3sgXpBrwAPNGLk5YCRTNTd"')
#     test_c = test_feature.query('purchase_id == "PMDTVvzExc6nMHnS8y5UV8"')
#     assert train_c['weekday'].values[0] == 1
#     assert test_c['weekday'].values[0] == 0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_date_info():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_date_info(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_date_info(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 2)
#     _base_shape_test(base_test, test_feature, 2)

#     # contents
#     train_c = train_feature.query('purchase_id == "3sgXpBrwAPNGLk5YCRTNTd"')
#     test_c = test_feature.query('purchase_id == "PMDTVvzExc6nMHnS8y5UV8"')
#     assert train_c['is_holiday'].values[0] == 0
#     assert train_c['is_holiday_eve'].values[0] == 0
#     assert test_c['is_holiday'].values[0] == 1
#     assert test_c['is_holiday_eve'].values[0] == 0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_purchase_time():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_purchase_time(
#         base_train, log, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_purchase_time(
#         base_test, log, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     train_c = train_feature.query('purchase_id == "njibeyLPrsnu4HCopjBihW"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['p_time'].values[0] == 14
#     assert test_c['p_time'].values[0] == 18


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_time_ap15():
#     base_train, base_test = get_base()
#     log = log_base.copy()
#     meta = meta_base.copy()
#     train_feature, _ = create_time_ap15(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_time_ap15(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     train_c = train_feature.query('purchase_id == "njibeyLPrsnu4HCopjBihW"')
#     test_c = test_feature.query('purchase_id == "C3rcdjjRyw9qSh6NcZMKSX"')
#     assert train_c['time_ap_15'].values[0] == 0
#     assert test_c['time_ap_15'].values[0] == 1


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_last_date_diff():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_last_date_diff(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_last_date_diff(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     train_c = train_feature.query('purchase_id == "dyqFNuhH2pJpHszpEciNKj"')
#     test_c = test_feature.query('purchase_id == "PMDTVvzExc6nMHnS8y5UV8"')
#     assert train_c['last_date_diff'].values[0] == 3.0
#     assert test_c['last_date_diff'].values[0] == 7.0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_1week_purchase_count():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_1week_purchase_count(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_1week_purchase_count(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     train_c = train_feature.query('purchase_id == "3sgXpBrwAPNGLk5YCRTNTd"')
#     test_c = test_feature.query('purchase_id == "PMDTVvzExc6nMHnS8y5UV8"')
#     assert train_c['1week_purchase_count'].values[0] == 1.0
#     assert test_c['1week_purchase_count'].values[0] == 0.0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_4week_purchase_count():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_4week_purchase_count(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_4week_purchase_count(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature)
#     _base_shape_test(base_test, test_feature)

#     # contents
#     train_c = train_feature.query('purchase_id == "3sgXpBrwAPNGLk5YCRTNTd"')
#     test_c = test_feature.query('purchase_id == "PMDTVvzExc6nMHnS8y5UV8"')
#     assert train_c['4week_purchase_count'].values[0] == 6.0
#     assert test_c['4week_purchase_count'].values[0] == 10.0


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_cd3_7days_mean():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_cd3_7days_mean(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_cd3_7days_mean(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 78)
#     _base_shape_test(base_test, test_feature, 78)

#     # contents
#     train_c = train_feature.query('purchase_id == "3sgXpBrwAPNGLk5YCRTNTd"')
#     test_c = test_feature.query('purchase_id == "PMDTVvzExc6nMHnS8y5UV8"')
#     assert train_c['cd3_7d_mean_110100'].values[0] == 0.42857142857142855
#     assert test_c['cd3_7d_mean_110100'].values[0] == 0.14285714285714285


# @pytest.mark.skipif(do_not_test, reason='skip in CI')
# def test_create_user_cd3_28days_mean():
#     base_train, base_test = get_base()
#     meta = meta_base.copy()
#     train_feature, _ = create_user_cd3_28days_mean(
#         base_train, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='train')
#     test_feature, _ = create_user_cd3_28days_mean(
#         base_test, meta, use_cache=use_cache, save_cache=save_cache, cache_prefix='test')

#     # basic shape test
#     _base_shape_test(base_train, train_feature, 78)
#     _base_shape_test(base_test, test_feature, 78)

#     # contents
#     train_c = train_feature.query('purchase_id == "3sgXpBrwAPNGLk5YCRTNTd"')
#     test_c = test_feature.query('purchase_id == "PMDTVvzExc6nMHnS8y5UV8"')
#     assert train_c['cd3_28d_mean_110100'].values[0] == 0.14285714285714285
#     assert test_c['cd3_28d_mean_110100'].values[0] == 0.1
