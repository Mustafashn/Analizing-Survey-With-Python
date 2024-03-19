import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
from statsmodels.stats.weightstats import ztest


def estimate_coef(x, y):
    # number of points
    n = np.size(x)

    # mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * mean_y * mean_x
    SS_xx = np.sum(x * x) - n * mean_x * mean_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx  # Eğim katsayısı, bağımsız değişkenin bağımlı değişken üzerindeki etkisini ölçer.
    # bağımsız değişkenin değişimiyle bağımlı değişkenin ne kadar değiştiğini ifade eder.
    b_0 = mean_y - b_1 * mean_x  # Kesim noktası, regresyon doğrusunun bağımsız değişken eksenini kestiği noktadır.

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the points
    plt.scatter(x, y, color="m", marker="o", s=30)
    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def T_test_assumptions(x, y):
    # first check the normally distributed
    norm_x = stats.shapiro(x)
    norm_y = stats.shapiro(y)
    if not (norm_x[1] >= 0.05 and norm_y[1] >= 0.05):
        print("Failed pre-tests beacuse of normal distribution")
    # check the equal variances
    var_test = stats.levene(x, y)
    if var_test[1] <= 0.05:
        print("Failed pre-tests beacuse of variance")
    print(stats.ttest_ind(x, y))


def perform_chi_square_tests(data, col1, col2):
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    significant = p < 0.05
    return significant


def calculate_z_score(data1, data2):
    data1_mean = np.mean(data1)
    data2_mean = np.mean(data2)
    data1_size = np.count_nonzero(data1)
    data2_size = np.count_nonzero(data2)

    # actually we can calculate std for datasets, but now we also know std's
    data1_std = 100
    data2_std = 90
    population_mean_difference = 10  # we know that also
    alpha = 0.05

    zscore = ((data1_mean - data2_mean) - population_mean_difference) / (
        math.sqrt((data1_std ** 2 / data1_size) + (data2_std ** 2 / data2_size)))
    critical_value = 1.645  # from z table
    print(zscore)
    if zscore < critical_value:
        print('Null hypothesis is accepted!')
    else:
        print('Null hypothesis is rejected. \nAlternate hypothesis is accepted!')


def main():
    #  REGRESSION
    """""
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    b = estimate_coef(x, y)
    print(f'Estimated coefficients:\n b_0 = {b[0]} \
         \n b_1 = {b[1]}')
    plot_regression_line(x,y,b)"""
    #  Two Sample T-TEST
    """""
    data_group1 = np.array([14, 15, 15, 16, 13, 8, 14,
                            17, 16, 14, 19, 20, 21, 15,
                            15, 16, 16, 13, 14, 12])

    data_group2 = np.array([15, 17, 14, 17, 14, 8, 12,
                            19, 19, 14, 17, 22, 24, 16,
                            13, 16, 13, 18, 15, 13])
    T_test_assumptions(data_group1,data_group2) """

    # Chi-square-test
    """""
    
    # reading dataset
    mat_data = pd.read_csv("student-mat.csv", sep=";")
    print(perform_chi_square_tests(mat_data,'G3','schoolsup'))
    """

    """""
    # Z-TEST
    #  Suppose we want to test if Girls on an average score 10 marks more than the boys. Suppose we also know that the
    #  standard deviation for girl’s Score is 100 and for boy’s score is 90. We collect the data of 20 girls and 20 boys
    #  by using random samples and record their marks. Finally, we also set our alpha (⍺) value (significance level) to
    #  be 0.05.
    sample1 = [650, 730, 510, 670, 480, 800, 690, 530, 590, 620, 710, 670, 640, 780, 650, 490, 800, 600, 510, 700]
    sample2 = [630, 720, 462, 631, 440, 783, 673, 519, 543, 579, 677, 649, 632, 768, 615, 463, 781, 563, 488, 650]
    calculate_z_score(sample1,sample2) """


main()
