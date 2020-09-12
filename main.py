import DataVisualizationUtils as dvu
import os
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import DataVisualizationUtils
from DataUtils import DataUtils
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    data_utils = DataUtils()

    # load the data from CSV file to pandas DataFrame (start from the project root directory)
    housing_data_frame = data_utils.load_csv_to_pandas_df(os.path.join("dataset", "housing", "housing.csv"));

    # Visualization of the data - geographically (by lat lang)
    DataVisualizationUtils.scatter_plot(housing_data_frame,
                                        "median_house_value",
                                        x_name="longitude",
                                        y_name="latitude",
                                        circle_radius=housing_data_frame["population"]/100,
                                        label="population size (Expressed by the circle radius)")

    # cross correlation matrix
    DataVisualizationUtils.cross_correlation_matrix(housing_data_frame);

    # scatter plot matrix
    attribute_scatter_plot_matrix = ["median_house_value", "median_income", "total_rooms", "housing_median_age"];
    DataVisualizationUtils.scatter_plot_matrix(housing_data_frame, attribute_scatter_plot_matrix);

    # Create more relevant/make sense new columns from the data which help us to predict
    # TODO: NEED TO MAKE IT AN ATTRIBUTE TRANSFORM
    housing_data_frame["rooms_per_household"] = housing_data_frame["total_rooms"] / housing_data_frame["households"];
    housing_data_frame["bedrooms_per_rooms"] = housing_data_frame["total_bedrooms"] / housing_data_frame["total_rooms"];
    housing_data_frame["population_per_household"] = housing_data_frame["population"] / housing_data_frame["households"];

    # By checking again the correlation matrix we can see that we created new more correlated feature with
    # the house prices.
    DataVisualizationUtils.cross_correlation_vector(housing_data_frame, "median_house_value")

    # split to train set and test set
    test_data_frame, train_data_frame = data_utils.split_test_train_set_by_stratefied_sampling(housing_data_frame, "median_income",
                                                                                 [0., 1.5, 3., 4.5, 6., np.inf]);

    # create label data frame
    train_labels_data_frame = data_utils.copy_and_drop_column(train_data_frame, "median_house_value")
    test_labels_data_frame = data_utils.copy_and_drop_column(test_data_frame, "median_house_value")


    # Prapare the data for ML algorithm
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scalar', StandardScaler()),
    ])

    numerical_attribute = list(test_data_frame.columns)
    numerical_attribute.remove("ocean_proximity");
    categorical_attribute = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_attribute),
        ('categorical', OneHotEncoder(), categorical_attribute)
    ])

    housing_train_data_prepared = full_pipeline.fit_transform(train_data_frame)
    housing_test_data_prepared = full_pipeline.fit_transform(test_data_frame)

    # training the algorithm
    linear_regression = LinearRegression();
    linear_regression.fit(housing_train_data_prepared, train_labels_data_frame);
    house_prices_prediction_result = linear_regression.predict(housing_test_data_prepared);
    mean_square_error = np.sqrt(mean_squared_error(house_prices_prediction_result, test_labels_data_frame));
    print(mean_square_error)

if __name__ == '__main__':
    main();


