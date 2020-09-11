import DataVisualizationUtils as dvu
import os
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from DataLoaderUtils import DataLoaderUtils
from pandas.plotting import scatter_matrix


def main():
    dataLoader = DataLoaderUtils()

    # load the data from CSV file to pandas DataFrame
    housingDataFrame = dataLoader.loadCsvToDataFrame(os.path.join("dataset", "housing", "housing.csv"));

    # split the data to test and train set, by stratified sampling.
    testSet, trainSet = dataLoader.splitToTestTrainSetByStratefiedSampling(housingDataFrame, "median_income", [0.,1.5,3.,4.5,6., np.inf]);

    # Visualization of the data - geographically (by lat lang)
    housing = trainSet.copy();
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population size (Expressed by the circle radius)", figsize=(15,10),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True);
    plt.legend();
    plt.show();

    # Trying to understand the features that most correlate with the house prices. We can calculate the
    # cross correlation between every pair of feature and tests the house prices specifically.
    # This correlation coefficient is called 'pearson's r'.
    corrMatrix = housing.corr();
    plt.figure(figsize=(10,10))
    sn.heatmap(corrMatrix, annot=True, vmin=-1, vmax=1)
    plt.show()

    # We can also to check the scatter matrix of all the possible features pair (or only part of them in this case)
    # and trying to understand if there is also a non-linear correlation between them (because the previous calculation
    # only can find a linear correlation between features).
    attribute = ["median_house_value", "median_income", "total_rooms", "housing_median_age"];
    scatter_matrix(housing[attribute], figsize=(15, 10))
    plt.show()

    stop = 0
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();


