import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer


class DataUtils:
    def __init__(self):
        pass

    def load_data_from_web(self, housingURL, housingPath):
        os.makedirs(housingPath, exist_ok=True);
        tgzPath = os.path.join(housingPath, "housing.csv");
        urllib.request.urlretrieve(housingURL, tgzPath);

    def load_csv_to_pandas_df(self, housingPath):
        """
        Read data from CSV file.
        :return: pandas DataFrame
        """
        return pd.read_csv(housingPath);

    def split_test_train_sets(self, dataFrame, testRatio=0.2):
        """
        Split the dataFrame to sets set and trian set in ratio of :testRation:
        :param dataFrame:
        :param testRatio:
        :return:
        """
        length = len(dataFrame);
        permutation = np.random.permutation(length);
        testSetSize = int(testRatio * length);
        testIndices = permutation[:testSetSize];
        trainIndices = permutation[testSetSize:];

        # take the specific indices for every set
        return dataFrame.iloc[testIndices], dataFrame.iloc[trainIndices];

    def split_test_train_set_by_stratefied_sampling(self, dataFrame, key, bins):
        """
        Suppose we know that the <key> have a significant effect about what we want to
        predict, and therefore we need to ensure that the test set is representative
        the various categories of <key> (that is we should to preserve the ration of the various
        <key> values in the whole dataset, in our test set).
        This manner of sampling is call 'stratified sampling', each category (bin that represent
        a range of values) called 'starta'. We split the whole data to homogenous and sampling each bin.
        In such a way we can avoid bias following wrong data sampling.

        :param dataFrame: Pandas DataFrame
        :param key:       column name we want to divide into categories
        :param bins:      Numpy array represent a ranges for each bin
        :return:
        """
        dataFrame['categories'] = pd.cut(dataFrame[key], bins=bins, labels=np.arange(1, len(bins)));
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42);
        (trainIndex, testIndex), = split.split(dataFrame, dataFrame['categories']);
        stratTestSet = (dataFrame.loc[testIndex]).drop('categories', axis=1);
        stratTrainSet = dataFrame.loc[trainIndex].drop('categories', axis=1);
        return stratTestSet, stratTrainSet;

    def copy_and_drop_column(self, dataFrame, key):
        label_df = pd.DataFrame({key: dataFrame[key].copy()});
        self.drop_col(dataFrame, key);
        return label_df;

    def drop_col(self, dataFrame, key):
        dataFrame.drop(key, axis=1, inplace=True);

    def fill_missing_values(self, dataFrame, drop):
        sk_imputer = SimpleImputer(strategy='median');
        non_numerical_col = self.copy_and_drop_column(dataFrame, drop);
        sk_imputer.fit(dataFrame);
        fixed_data = sk_imputer.transform(dataFrame); # return a numpy array
        dataFrame = pd.DataFrame(fixed_data, columns=dataFrame.columns, index=dataFrame.index);
        dataFrame[drop] = non_numerical_col
        print(dataFrame.info());
        return dataFrame;



