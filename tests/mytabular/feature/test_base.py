import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from mytabular.feature.base import Feature, FEATURE_DIR


class SampleFeature(Feature):
    def create(self, base, others=None):
        return base


class TestFeatureBase:

    def test_init(self):
        feature_train = SampleFeature(name='sample_feature', train=True, category='sample_competition')
        assert feature_train.name == 'sample_feature'
        assert feature_train.train
        assert feature_train.name_prefix == 'train'
        assert feature_train.category == 'sample_competition'

        feature_test = SampleFeature(name='sample_feature', train=False, category='sample_competition')
        assert feature_test.name == 'sample_feature'
        assert not feature_test.train
        assert feature_test.name_prefix == 'test'
        assert feature_test.category == 'sample_competition'

    def test_path(self):
        feature_train = SampleFeature(name='sample_feature', train=True, category='sample_competition')
        feature_test = SampleFeature(name='sample_feature', train=False, category='sample_competition')

        assert str(feature_train._path) == str(FEATURE_DIR / 'sample_competition' / 'train' / 'sample_feature.ftr')
        assert str(feature_test._path) == str(FEATURE_DIR / 'sample_competition' / 'test' / 'sample_feature.ftr')

    def test_save_and_load(self):
        sample_df = pd.DataFrame({'feature1': [0, 1, 2, 3], 'feature2': np.random.randn(4)})
        feature_train = SampleFeature(name='sample_feature', train=True, category='sample_competition')
        feature = feature_train(sample_df, save_cache=True)
        assert np.all(sample_df == feature)

        # test save and load
        feature = feature_train(sample_df, use_cache=True)
        assert np.all(sample_df == feature)

        # create() is not called when use_cache
        mock = MagicMock()
        with patch.object(SampleFeature, 'create', mock):
            feature = feature_train(sample_df, use_cache=True)
            assert np.all(sample_df == feature)
            assert not mock.called
