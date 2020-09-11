import pandas as pd
import matplotlib.pyplot as plt

def histogramOf(dataFrame, bins):
    dataFrame.hist(bins=50, figsize=(20, 15));
    plt.show();

def descriveData(dataFrame):
    dataFrame.describe();