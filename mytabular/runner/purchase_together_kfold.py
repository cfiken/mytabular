# from typing import List
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score
# import lightgbm

# import report
# from mytabular.runner.base import BaseRunner, Result
# from mytabular.feature.feature import (
#     create_user_label,
#     create_shop_label,
#     create_purchase_together_cd1,
#     create_purchase_together_cd2,
#     create_purchase_together_cd3,
#     create_purchase_together_cd4,
#     create_purchase_date,
#     create_weekday,
#     create_date_info,
#     create_purchase_time,
#     create_time_ap15,
#     create_last_date_diff,
#     create_1week_purchase_count,
#     create_4week_purchase_count
# )

# # train = pd.read_csv('/home/td009/kaggle-toguro/data/train.csv')
# # test = pd.read_csv('/home/td009/kaggle-toguro/data/test.csv')
# # meta = pd.read_csv('/home/td009/kaggle-toguro/data/meta.csv')
# # log = pd.read_csv('/home/td009/kaggle-toguro/data/purchase_log.csv')
# # category = pd.read_csv('/home/td009/kaggle-toguro/data/category.csv')
# # jan = pd.read_csv('/home/td009/kaggle-toguro/data/jan.csv')
# # submission = pd.read_csv('/home/td009/kaggle-toguro/data/sample_submission.csv')

# # idx_to_column = {k: v for k, v in enumerate(list(submission.columns))}

# lgb_params = {
#     'objective': 'binary',
#     'boosting': 'gbdt',
#     'learning_rate': 0.1,
#     'num_leaves': 31,
#     'seed': 1019,
#     'max_depth': -1,
#     'min_child_samples': 10,
# }


# class PurchaseTogetherRunner(BaseRunner):

#     def run(self):
#         print('# start!!')
#         # config
#         self.num_splits = 5

#         train_target = train.copy().set_index('purchase_id')
#         base_train, base_test = train.copy()[['purchase_id']], test.copy()[['purchase_id']]
#         print('## creating train feature')
#         train_feature = self.create_feature(base_train, train=True)
#         print('## creating test feature')
#         test_feature = self.create_feature(base_test, train=False)
#         print('## splitting data')
#         splits = self.split_validation(train_feature, train_target, num_splits=self.num_splits)
#         print('## training')
#         result = self.train(splits, train_feature, train_target, self.num_splits)
#         print('### valdiation result')
#         print(result.metrics)

#         print('## predicting')
#         pred = self.predict(result.models, test_feature, len(idx_to_column))
#         self.submit(pred)
#         self.notify(result)
#         print('# finish!!')

#     def create_feature(self, base: pd.DataFrame, train: bool):
#         feature = base
#         # feature = create_user_label(feature, meta.copy())
#         feature = create_shop_label(feature, meta.copy())
#         feature = create_purchase_together_cd1(feature, log.copy(), category.copy())
#         feature = create_purchase_together_cd2(feature, log.copy(), category.copy())
#         feature = create_purchase_together_cd3(feature, log.copy(), category.copy())
#         # feature = create_purchase_together_cd4(feature, log.copy(), category.copy())
#         feature = create_last_date_diff(feature, meta.copy())
#         feature = create_purchase_time(feature, log.copy(), meta.copy())
#         feature = create_1week_purchase_count(feature, meta.copy())
#         feature = create_4week_purchase_count(feature, meta.copy())
#         feature = feature.set_index('purchase_id')
#         return feature

#     def split_validation(self,
#                          feature: pd.DataFrame,
#                          target: pd.DataFrame,
#                          num_splits: int = 5):
#         splitter = KFold(num_splits, shuffle=True, random_state=1019)
#         splits = splitter.split(feature, y=target)
#         return splits

#     def train(self,
#               splits,
#               feature: pd.DataFrame,
#               target: pd.DataFrame,
#               num_splits: int) -> List[Result]:
#         results = []
#         lgb_models = []
#         valid_preds = np.zeros((target.shape[0], len(target.columns)), dtype=np.float32)
#         aucs = np.zeros(len(target.columns))

#         for i, (train_idx, valid_idx) in enumerate(splits):
#             print(f'fold start {i}')
#             train_x = feature.values[train_idx]
#             valid_x = feature.values[valid_idx]
#             train_y = target.values[train_idx]
#             valid_y = target.values[valid_idx]

#             result = self._train_fold(train_x,
#                                       train_y,
#                                       valid_x,
#                                       valid_y,
#                                       num_classes=target.shape[1],
#                                       fold=i)
#             valid_preds[valid_idx, :] = result.oof_preds
#             lgb_models.append(result.models)
#             aucs += result.metrics['aucs'] / num_splits
#         metrics = {'auc': np.mean(aucs), 'aucs': aucs}

#         return Result(oof_preds=valid_preds, models=lgb_models, metrics=metrics)


#     def _train_fold(self,
#                     train_x: np.ndarray,
#                     train_y: np.ndarray,
#                     valid_x: np.ndarray,
#                     valid_y: np.ndarray,
#                     num_classes: int,
#                     fold: int):
#         fold_models = []
#         for i in range(num_classes):
#             train_dataset = lightgbm.Dataset(train_x, train_y[:, i])
#             valid_dataset = lightgbm.Dataset(valid_x, valid_y[:, i])
#             model = lightgbm.train(lgb_params,
#                                    train_dataset,
#                                    num_boost_round=1000,
#                                    valid_sets=[valid_dataset, train_dataset],
#                                    valid_names=['valid', 'train'],
#                                    early_stopping_rounds=100)
#             fold_models.append(model)

#         fold_aucs = []
#         valid_preds = np.zeros((len(valid_x), num_classes))
#         for i in range(num_classes):
#             model = fold_models[i]
#             pred = model.predict(valid_x)
#             valid_preds[:, i] = pred
#             fold_aucs.append(roc_auc_score(valid_y[:, i], pred))
#         metrics = {'aucs': np.array(fold_aucs)}
#         return Result(oof_preds=valid_preds, models=fold_models, metrics=metrics)

#     def predict(self,
#                 models: List,
#                 feature: pd.DataFrame,
#                 num_classes: int = 13) -> pd.DataFrame:
#         preds = {}
#         for i in range(num_classes):
#             preds[idx_to_column[i]] = np.zeros(len(feature))
#             for j in range(len(models)):
#                 model = models[j][i]
#                 pred = model.predict(feature)
#                 preds[idx_to_column[i]] += pred / len(models)
#         return pd.DataFrame(preds)

#     def submit(self, pred: pd.DataFrame) -> None:
#         pred.to_csv('submission.csv', index=False)

#     def notify(self, result: Result):
#         cv = result.metrics['auc']
#         description = self.description
#         description += '\n' + 'cv: ' + str(result.metrics['aucs'])
#         report.report_success(self.name, cv, description, {})
