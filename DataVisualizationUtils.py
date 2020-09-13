import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pandas.plotting import scatter_matrix


def histogramOf(dataFrame, bins):
    dataFrame.hist(bins=bins, figsize=(20, 15));
    plt.show();


def scatter_plot(data_frame, colored_by_attribute, x_name='x', y_name='y', circle_radius=1, label='circle label'):
    """
    :param data_frame:            pandas DataFrame
    :param colored_by_attribute:  the circles colors scale will reflect the <colored_by_attribute> values
    :param x_name:                axis x name
    :param y_name:                axis y name
    :param circle_radius:         a measure by which the radius of the circles is determined
    :param label:                 label of the circles
    :return: void
    """
    data_frame.plot(kind="scatter", x=x_name, y=y_name, alpha=0.4, s=circle_radius,
                    label=label, figsize=(15, 10), c=colored_by_attribute,
                    cmap=plt.get_cmap("jet"), colorbar=True);
    plt.legend();
    plt.show();


def cross_correlation_matrix(data_frame):
    """
    Trying to find the features that most correlate with each other. We can calculate the
    cross correlation between every pair of features (in case there is so much of them)
    and tests the house the features that we are most interested.
    This correlation coefficient is called 'pearson's r'.
    :return: void
    """

    plt.figure(figsize=(15, 15))
    sn.heatmap(data_frame.corr(), annot=True, vmin=-1, vmax=1)
    plt.show()


def cross_correlation_vector(data_frame, attribute):
    """
    :param data_frame: pandas DataFrame
    :param attribute:  the attribute that we display the cross correlation between it with the other attributes
    :return: void
    """

    plt.figure(figsize=(5, 10))
    cross_vector = data_frame.corr()[[attribute]].sort_values(ascending=False, by=attribute);
    sn.heatmap(cross_vector, annot=True, vmin=-1, vmax=1)
    plt.show()


def scatter_plot_matrix(data_frame, attributes):
    """
    We can also to check the scatter matrix of all the possible features pair (or only part of them)
    and trying to understand if there is also a non-linear correlation between them (because
    <cross_correlation_matrix> only can find a linear correlation between features).

    :param data_frame:
    :param attributes:
    :return:
    """

    scatter_matrix(data_frame[attributes], figsize=(15, 10))
    plt.show()


def describe_data(data_frame):
    data_frame.describe();

def print_with_title(title, body):
    print("====================================================");
    print(title, ": ", '\n', body, '\n');
