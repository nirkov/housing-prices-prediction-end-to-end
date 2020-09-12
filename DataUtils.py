import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer


class DataUtils:
    def __init__(self):
        pass

    @staticmethod
    def load_data_from_web(self, data_url, data_local_path):
        os.makedirs(data_local_path, exist_ok=True);
        tgzPath = os.path.join(data_local_path, "housing.csv");
        urllib.request.urlretrieve(data_url, tgzPath);

    @staticmethod
    def load_csv_to_pandas_df(housingPath):
        """
        Read data from CSV file.
        :return: pandas DataFrame
        """
        return pd.read_csv(housingPath);

    @staticmethod
    def split_test_train_sets(data_frame, test_ratio=0.2):
        """
        Split the dataFrame to sets set and trian set in ratio of :testRation:
        :param data_frame:
        :param test_ratio:
        :return:
        """
        length = len(data_frame);
        permutation = np.random.permutation(length);
        testSetSize = int(test_ratio * length);
        testIndices = permutation[:testSetSize];
        trainIndices = permutation[testSetSize:];

        # take the specific indices for every set
        return data_frame.iloc[testIndices], data_frame.iloc[trainIndices];

    @staticmethod
    def split_test_train_set_by_stratified_sampling(data_frame, key, bins):
        """
        Suppose we know that the <key> have a significant effect about what we want to
        predict, and therefore we need to ensure that the test set is representative
        the various categories of <key> (that is we should to preserve the ration of the various
        <key> values in the whole dataset, in our test set).
        This manner of sampling is call 'stratified sampling', each category (bin that represent
        a range of values) called 'starta'. We split the whole data to homogenous and sampling each bin.
        In such a way we can avoid bias following wrong data sampling.

        :param data_frame: Pandas DataFrame
        :param key:       column name we want to divide into categories
        :param bins:      Numpy array represent a ranges for each bin
        :return:
        """
        temp_column_name = 'categories';
        data_frame[temp_column_name] = pd.cut(data_frame[key], bins=bins, labels=np.arange(1, len(bins)));
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42);
        (trainIndex, testIndex), = split.split(data_frame, data_frame[temp_column_name]);
        stratTestSet = (data_frame.loc[testIndex]).drop(temp_column_name, axis=1);
        stratTrainSet = data_frame.loc[trainIndex].drop(temp_column_name, axis=1);
        return stratTestSet, stratTrainSet;

    @staticmethod
    def copy_and_drop_column(data_frame, key):
        label_df = pd.DataFrame({key: data_frame[key].copy()});
        DataUtils.drop_col(data_frame, key);
        return label_df;

    @staticmethod
    def drop_col(data_frame, key):
        data_frame.drop(key, axis=1, inplace=True);

    @staticmethod
    def fill_missing_values(data_frame, drop):
        sk_imputer = SimpleImputer(strategy='median');
        non_numerical_col = DataUtils.copy_and_drop_column(data_frame, drop);
        sk_imputer.fit(data_frame);
        fixed_data = sk_imputer.transform(data_frame); # return a numpy array
        dataFrame = pd.DataFrame(fixed_data, columns=data_frame.columns, index=data_frame.index);
        dataFrame[drop] = non_numerical_col
        print(dataFrame.info());
        return dataFrame;



