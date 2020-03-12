from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm

from toguro.lib import report
from cfiken.atma4.runner.base import BaseRunner, Result
from cfiken.atma4.feature.feature import (
    create_user_label,
    create_shop_label,
    create_target_in_train,
    create_purchase_together_cd1,
    create_purchase_together_cd2,
    create_purchase_together_cd3,
    create_purchase_together_cd4,
    create_purchase_together_cd3_pca4,
    create_purchase_together_cd3_pca8,
    create_purchase_together_cd3_pca16,
    create_purchase_together_cd3_tfidf_pca4,
    create_purchase_together_cd3_tfidf_pca8,
    create_purchase_together_cd3_tfidf_pca16,
    create_purchase_together_cd4_pca4,
    create_purchase_together_cd4_pca8,
    create_purchase_together_cd4_pca16,
    create_purchase_together_cd4_tfidf_pca4,
    create_purchase_together_cd4_tfidf_pca8,
    create_purchase_together_cd4_tfidf_pca16,
    create_user_purchase_mean_cd2,
    create_user_purchase_mean_cd3,
    create_user_purchase_sum_cd4_pca4,
    create_user_purchase_sum_cd4_pca8,
    create_user_purchase_sum_cd4_pca16,
    create_user_purchase_sum_cd4_tfidf_pca4,
    create_user_purchase_sum_cd4_tfidf_pca8,
    create_user_purchase_sum_cd4_tfidf_pca16,
    create_user_purchase_mean_cd4_pca4,
    create_user_purchase_mean_cd4_pca8,
    create_user_purchase_mean_cd4_pca16,
    create_user_purchase_mean_cd4_tfidf_pca4,
    create_user_purchase_mean_cd4_tfidf_pca8,
    create_user_purchase_mean_cd4_tfidf_pca16,
    create_purchase_date,
    create_weekday,
    create_date_info,
    create_purchase_time,
    create_time_ap15,
    create_last_date_diff,
    create_1week_purchase_count,
    create_4week_purchase_count,
    create_user_cd3_7days_mean,
    create_user_cd3_28days_mean
)
from cfiken.atma4.util import save_feature_importance, SEED

train = pd.read_csv('/home/td009/kaggle-toguro/data/train.csv')
test = pd.read_csv('/home/td009/kaggle-toguro/data/test.csv')
meta = pd.read_csv('/home/td009/kaggle-toguro/data/meta.csv')
log = pd.read_csv('/home/td009/kaggle-toguro/data/purchase_log.csv')
category = pd.read_csv('/home/td009/kaggle-toguro/data/category.csv')
jan = pd.read_csv('/home/td009/kaggle-toguro/data/jan.csv')
submission = pd.read_csv('/home/td009/kaggle-toguro/data/sample_submission.csv')

idx_to_column = {k: v for k, v in enumerate(list(submission.columns))}

lgb_params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': SEED,
    'max_depth': -1,
    'min_child_samples': 10,
    'metric': 'auc',
    'importance_type': 'gain'
}


class PurchasePCARunner(BaseRunner):

    def run(self):
        print('# start!!')
        # config
        self.num_splits = 5

        train_target = train.copy().set_index('purchase_id')
        base_train, base_test = train.copy()[['purchase_id']], test.copy()[['purchase_id']]
        print('## creating feature')
        train_feature = self.create_feature(base_train, use_cache=True, cache_prefix='train')
        test_feature = self.create_feature(base_test, use_cache=True, cache_prefix='test')
        train_unsupervised, test_unsupervised = self.create_feature_both(base_train, base_test, use_cache=True)
        train_feature = pd.merge(train_feature, train_unsupervised, how='inner', on='purchase_id')
        test_feature = pd.merge(test_feature, test_unsupervised, how='inner', on='purchase_id')
        print('## splitting data')
        splits = self.split_validation(train_feature, train_target, num_splits=self.num_splits)
        print('## training')
        result = self.train(splits, train_feature, train_target, self.num_splits)
        print('### validation result')
        print(result.metrics)

        print('## predicting')
        pred = self.predict(result.models, test_feature, len(idx_to_column))
        self.submit(pred)
        self.notify(result)
        print('# finish!!')

    def create_feature(self, base: pd.DataFrame, use_cache: bool, cache_prefix: str):
        feature = base.copy()
        # feature = create_user_label(feature, meta.copy())
        feature = self._feature_wrapper(create_shop_label(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        # feature = self._feature_wrapper(create_target_in_train(
        #     feature, meta.copy(), log.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_purchase_together_cd1(
            feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_purchase_together_cd2(
            feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_purchase_together_cd3(
            feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_user_purchase_mean_cd2(
            feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_user_purchase_mean_cd3(
            feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        # feature = self._feature_wrapper(create_purchase_together_cd4(
        #     feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_last_date_diff(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_weekday(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_date_info(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_purchase_time(
            feature, log.copy(), meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_time_ap15(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_1week_purchase_count(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_4week_purchase_count(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_user_cd3_7days_mean(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_user_cd3_28days_mean(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = feature.set_index('purchase_id')
        return feature

    def create_feature_both(self,
                            base_train: pd.DataFrame,
                            base_test: pd.DataFrame,
                            use_cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        train と test 両方が同時に必要な特徴を作成する。
        '''
        def merge_features(train, test, tmp_train, tmp_test):
            train = pd.merge(train, tmp_train, how='inner', on='purchase_id')
            test = pd.merge(test, tmp_test, how='inner', on='purchase_id')
            return train, test

        train_feature, test_feature = base_train.copy(), base_test.copy()
        # _train, _test = self._feature_wrapper(create_purchase_together_cd3_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        # _train, _test = self._feature_wrapper(create_purchase_together_cd3_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        # _train, _test = self._feature_wrapper(create_purchase_together_cd3_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        # _train, _test = self._feature_wrapper(create_purchase_together_cd3_tfidf_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        # _train, _test = self._feature_wrapper(create_purchase_together_cd3_tfidf_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        # _train, _test = self._feature_wrapper(create_purchase_together_cd3_tfidf_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_pca32(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_tfidf_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_tfidf_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_tfidf_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_purchase_together_cd4_tfidf_pca32(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        _train, _test = self._feature_wrapper(create_user_purchase_mean_cd4_pca16(
            base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_tfidf_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_tfidf_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        _train, _test = self._feature_wrapper(create_user_purchase_sum_cd4_tfidf_pca16(
            base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        train_feature, test_feature = merge_features(train_feature, test_feature, _train, _test)
        train_feature = train_feature.set_index('purchase_id')
        test_feature = test_feature.set_index('purchase_id')
        return train_feature, test_feature

    def split_validation(self,
                         feature: pd.DataFrame,
                         target: pd.DataFrame,
                         num_splits: int = 5):
        splitter = KFold(num_splits, shuffle=True, random_state=1019)
        splits = splitter.split(feature, y=target)
        return splits

    def train(self,
              splits,
              feature: pd.DataFrame,
              target: pd.DataFrame,
              num_splits: int) -> List[Result]:
        lgb_models = []
        valid_preds = np.zeros((target.shape[0], len(target.columns)), dtype=np.float32)
        aucs = np.zeros(len(target.columns))

        for i, (train_idx, valid_idx) in enumerate(splits):
            print(f'fold start {i}')
            train_x = feature.values[train_idx]
            valid_x = feature.values[valid_idx]
            train_y = target.values[train_idx]
            valid_y = target.values[valid_idx]

            result = self._train_fold(train_x,
                                      train_y,
                                      valid_x,
                                      valid_y,
                                      num_classes=target.shape[1],
                                      fold=i)
            valid_preds[valid_idx, :] = result.oof_preds
            lgb_models.append(result.models)
            aucs += result.metrics['aucs'] / num_splits
            save_feature_importance(result.models[-1].feature_importance(importance_type='gain'),
                                    feature.columns,
                                    self.outputdir,
                                    suffix=str(i))
        metrics = {'auc': np.mean(aucs), 'aucs': aucs}

        return Result(oof_preds=valid_preds, models=lgb_models, metrics=metrics)

    def _train_fold(self,
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    valid_x: np.ndarray,
                    valid_y: np.ndarray,
                    num_classes: int,
                    fold: int):
        fold_models = []
        for i in range(num_classes):
            train_dataset = lightgbm.Dataset(train_x, train_y[:, i])
            valid_dataset = lightgbm.Dataset(valid_x, valid_y[:, i])
            model = lightgbm.train(lgb_params,
                                   train_dataset,
                                   num_boost_round=1000,
                                   valid_sets=[valid_dataset, train_dataset],
                                   valid_names=['valid', 'train'],
                                   early_stopping_rounds=200)
            fold_models.append(model)

        fold_aucs = []
        valid_preds = np.zeros((len(valid_x), num_classes))
        for i in range(num_classes):
            model = fold_models[i]
            pred = model.predict(valid_x)
            valid_preds[:, i] = pred
            fold_aucs.append(roc_auc_score(valid_y[:, i], pred))
        metrics = {'aucs': np.array(fold_aucs)}
        return Result(oof_preds=valid_preds, models=fold_models, metrics=metrics)

    def predict(self,
                models: List,
                feature: pd.DataFrame,
                num_classes: int = 13) -> pd.DataFrame:
        preds = {}
        for i in range(num_classes):
            preds[idx_to_column[i]] = np.zeros(len(feature))
            for j in range(len(models)):
                model = models[j][i]
                pred = model.predict(feature)
                preds[idx_to_column[i]] += pred / len(models)
        return pd.DataFrame(preds)

    def notify(self, result: Result):
        cv = result.metrics['auc']
        description = self.description
        description += '\n' + 'cv: ' + str(result.metrics['aucs'])
        params = {'features': ', '.join(set(self.feature_names))}
        report.report_success(self.name, cv, description, params=params)
