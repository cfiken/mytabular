from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm

from toguro.lib import report
from cfiken.atma4.runner.base import BaseRunner, Result
from cfiken.atma4.feature.feature import (
    create_user_label,
    create_shop_label,
    create_purchase_together_cd1,
    create_purchase_together_cd2,
    create_purchase_together_cd3,
    create_purchase_together_cd4,
    create_purchase_together_cd4_pca4,
    create_purchase_together_cd4_pca8,
    create_purchase_together_cd4_pca16,
    create_purchase_together_cd4_tfidf_pca4,
    create_purchase_together_cd4_tfidf_pca8,
    create_purchase_together_cd4_tfidf_pca16,
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
    create_4week_purchase_count
)
from cfiken.atma4.util import save_feature_importance

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
    'learning_rate': 0.2,
    'num_leaves': 31,
    'seed': 1019,
    'max_depth': -1,
    'min_child_samples': 10,
    'metric': 'auc',
    'importance_type': 'gain'
}


class AdversarialValidationRunner(BaseRunner):

    def run(self):
        print('# start!!')
        # config
        self.num_splits = 5

        train_target = self.create_target(train.copy(), 1)
        test_target = self.create_target(test.copy(), 0)
        all_target = pd.concat([train_target, test_target], axis=0)

        base_train, base_test = train.copy()[['purchase_id']], test.copy()[['purchase_id']]
        print('## creating feature')
        train_feature = self.create_feature(base_train, use_cache=True, cache_prefix='train')
        test_feature = self.create_feature(base_test, use_cache=True, cache_prefix='test')
        # train_unsupervised, test_unsupervised = self.create_feature_both(base_train, base_test, use_cache=True)
        # train_feature = pd.merge(train_feature, train_unsupervised, how='inner', on='purchase_id')
        # test_feature = pd.merge(test_feature, test_unsupervised, how='inner', on='purchase_id')
        all_feature = pd.concat([train_feature, test_feature], axis=0)
        print('## splitting data')
        splits = self.split_validation(all_feature, all_target, num_splits=self.num_splits)
        print('## training')
        result = self.train(splits, all_feature, all_target, self.num_splits)
        print('### validation result')
        print(result.metrics)

        print('## predicting')
        # pred = self.predict(result.models, test_feature, len(idx_to_column))
        # self.submit(pred)
        self.notify(result)
        print('# finish!!')

    def create_feature(self, base: pd.DataFrame, use_cache: bool, cache_prefix: str):
        feature = base.copy()
        # feature = create_user_label(feature, meta.copy())
        feature = self._feature_wrapper(create_shop_label(
            feature, meta.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        # feature = self._feature_wrapper(create_purchase_together_cd1(
        #     feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_purchase_together_cd2(
            feature, log.copy(), category.copy(), use_cache=use_cache, cache_prefix=cache_prefix))
        feature = self._feature_wrapper(create_purchase_together_cd3(
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
        feature = feature.set_index('purchase_id')
        return feature

    def create_feature_both(self,
                            base_train: pd.DataFrame,
                            base_test: pd.DataFrame,
                            use_cache: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        train と test 両方が同時に必要な特徴を作成する。
        '''
        train_feature, test_feature = base_train.copy(), base_test.copy()
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
        train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_pca4(
            base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_tfidf_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_tfidf_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_sum_cd4_tfidf_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_tfidf_pca4(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_tfidf_pca8(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        # train_feature, test_feature = self._feature_wrapper(create_user_purchase_mean_cd4_tfidf_pca16(
        #     base_train.copy(), base_test.copy(), log.copy(), category.copy(), meta.copy(), use_cache=use_cache))
        train_feature = train_feature.set_index('purchase_id')
        test_feature = test_feature.set_index('purchase_id')
        return train_feature, test_feature

    def create_target(self,
                      df: pd.DataFrame,
                      target: int) -> pd.DataFrame:
        df_target = df.copy()
        df_target['target'] = target
        df_target = df_target[['purchase_id', 'target']]
        df_target = df_target.set_index('purchase_id')
        return df_target

    def split_validation(self,
                         feature: pd.DataFrame,
                         target: pd.DataFrame,
                         num_splits: int = 5):
        splitter = StratifiedKFold(num_splits, shuffle=True, random_state=1019)
        splits = splitter.split(feature, y=target)
        return splits

    def train(self,
              splits,
              feature: pd.DataFrame,
              target: pd.DataFrame,
              num_splits: int) -> List[Result]:
        lgb_models = []
        valid_preds = np.zeros((target.shape[0]), dtype=np.float32)
        auc = 0.0

        for i, (train_idx, valid_idx) in enumerate(splits):
            print(f'fold start {i}')
            train_x = feature.values[train_idx]
            valid_x = feature.values[valid_idx]
            train_y = target.values[train_idx, 0]
            valid_y = target.values[valid_idx, 0]

            result = self._train_fold(train_x,
                                      train_y,
                                      valid_x,
                                      valid_y,
                                      fold=i)
            valid_preds[valid_idx] = result.oof_preds
            lgb_models.append(result.models)
            auc += result.metrics['auc'] / num_splits
            save_feature_importance(result.models.feature_importance(importance_type='gain'),
                                    feature.columns,
                                    self.outputdir,
                                    suffix=str(i))
        metrics = {'auc': np.mean(auc)}

        return Result(oof_preds=valid_preds, models=lgb_models, metrics=metrics)

    def _train_fold(self,
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    valid_x: np.ndarray,
                    valid_y: np.ndarray,
                    fold: int):
        train_dataset = lightgbm.Dataset(train_x, train_y)
        valid_dataset = lightgbm.Dataset(valid_x, valid_y)
        model = lightgbm.train(lgb_params,
                               train_dataset,
                               num_boost_round=200,
                               valid_sets=[valid_dataset, train_dataset],
                               valid_names=['valid', 'train'],
                               early_stopping_rounds=20)

        valid_preds = np.zeros((len(valid_x)))
        valid_pred = model.predict(valid_x)
        auc = roc_auc_score(valid_y, valid_pred)
        metrics = {'auc': np.array(auc)}
        return Result(oof_preds=valid_preds, models=model, metrics=metrics)

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
        description += '\n' + 'cv: ' + str(result.metrics['auc'])
        params = {'features': ', '.join(set(self.feature_names))}
        report.report_success(self.name, cv, description, params=params)
