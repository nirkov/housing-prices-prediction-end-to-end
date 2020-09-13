import DataVisualizationUtils as dvu
import numpy as np
import os
import DataVisualizationUtils
from DataUtils import DataUtils

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


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

    # By re-examination of the correlation matrix, we can see that we created new features that more correlated with
    # house prices.
    DataVisualizationUtils.cross_correlation_vector(housing_data_frame, "median_house_value")

    # split to train set and test set using stratified sampling
    test_data_frame, train_data_frame = data_utils.split_test_train_set_by_stratified_sampling(housing_data_frame, "median_income",
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


    housing_prepared = full_pipeline.fit_transform(housing_data_frame);
    housing_train_data_prepared = housing_prepared.take(train_data_frame.index, axis=0);
    housing_test_data_prepared = housing_prepared.take(test_data_frame.index, axis=0);


    # training the algorithm
    linear_regression = LinearRegression();
    linear_regression.fit(housing_train_data_prepared, train_labels_data_frame);
    prediction_result_linear_regression = linear_regression.predict(housing_test_data_prepared);
    mean_square_error_linear_regression = np.sqrt(mean_squared_error(prediction_result_linear_regression, test_labels_data_frame));
    DataVisualizationUtils.print_with_title("Linear Regression MSE", mean_square_error_linear_regression)

    # Trying decision tree model in case our data contain alot of non-linear correlation between the features
    tree_regression = DecisionTreeRegressor();
    tree_regression.fit(housing_train_data_prepared, train_labels_data_frame);
    prediction_result_decision_tree = tree_regression.predict(housing_test_data_prepared);
    mean_square_error_decision_tree = np.sqrt(mean_squared_error(prediction_result_decision_tree, test_labels_data_frame));
    DataVisualizationUtils.print_with_title("Decision Tree MSE", mean_square_error_decision_tree)

    # Trying cross validation
    scores = cross_val_score(DecisionTreeRegressor(), housing_train_data_prepared, train_labels_data_frame, scoring="neg_mean_squared_error", cv=10);
    scores = np.sqrt(-scores) # sklearn uses utility function rather than cost function so the results are negative.
    # print(scores);
    DataVisualizationUtils.print_with_title("Decision Tree Cross Validation MSE", scores.mean())
    # print(scores.std()); # 2566.8761488982286

    # Trying random forest
    random_forest = RandomForestRegressor();
    random_forest.fit(housing_train_data_prepared, train_labels_data_frame.values.ravel());
    prediction_result_random_forest = random_forest.predict(housing_test_data_prepared);
    mean_square_error_random_forest = np.sqrt(mean_squared_error(prediction_result_random_forest, test_labels_data_frame));
    DataVisualizationUtils.print_with_title("Rnadom Forest MSE", mean_square_error_random_forest)


    # Trying grid search
    param_grid_search = [
        {'n_estimators':[3, 10, 30], 'max_features': [2, 4, 6 ,8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ];

    random_forest_grid_search = RandomForestRegressor();
    grid_search_results = GridSearchCV(random_forest_grid_search, param_grid_search, cv=5, scoring="neg_mean_squared_error", return_train_score=True);
    grid_search_results.fit(housing_train_data_prepared, train_labels_data_frame.values.ravel());
    cv_results = grid_search_results.cv_results_;
    DataVisualizationUtils.print_with_title("Rnadom Forest with Grid Search MSE", "")
    for mean_score, param in zip(cv_results['mean_test_score'], cv_results['params']):
        print(np.sqrt(-mean_score), param)

if __name__ == '__main__':
    main();


