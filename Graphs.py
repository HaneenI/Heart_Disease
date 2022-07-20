# Importing libs and csv file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import collections
import statistics as stat
import joblib
import random
import scipy.stats as stats
import math
from scipy.stats import ttest_1samp
from sklearn.preprocessing import StandardScaler
import math
from tkinter import *
from PIL import ImageTk, Image

# scikit-learn lib for linear regression and splitting test/model data
from sklearn.model_selection import train_test_split  # Splitting data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy.stats import norm

class Graghs_and_predictions:
   def __init__(self):
       self.data_frame =pd.read_csv('heart2.csv')
       self.data_Age = self.data_frame.loc[:, 'Age']

   # Spliting DataSet
   # def spliting(self):
   #     # x with categorical data which we cant deal with
   #     x = self.data_frame.drop('HeartDisease', axis=1)
   #     # y only helds Heart Disease
   #     y =self.data_frame.HeartDisease
   #     # modified x terminates all categorical data and make 'dummies'-you may google this-
   #     modified_x = pd.get_dummies(x)
   #     modified_x.to_csv("meow.csv", index=False)

   def draw_bar_chart(my_plt, x_axis, y_axis, x_label):
       my_plt.bar(x_axis, y_axis, color='#ffa9a8', width=0.7, label=x_label, edgecolor="#ffea94")
       my_plt.xlabel(x_label)
       my_plt.ylabel('Number of people')
       my_plt.title("Number of people have " + x_label)
       my_plt.legend()
       my_plt.show()

   def draw_pie_chart(my_plt, labels, frequency, title):
       color = ["#ffa9a8", "#ffea94", "#a5cbef", "#5cc0bb"]
       my_plt.pie(frequency, labels=labels, autopct="%.1f%%", colors=color, shadow=True)
       my_plt.title("Number of people have " + title)
       my_plt.legend(title=title)
       my_plt.show()

   def draw_histogram(my_plt, my_data, title):
       x_axis = my_data.loc[:, title]
       median = x_axis.median()
       mean = x_axis.mean()
       mode = stat.mode(x_axis)

       my_plt.hist(x_axis, edgecolor='#ffea94', color='#ffa9a8')
       my_plt.axvline(median, color='#d9534f', label='median', linewidth=3)
       my_plt.axvline(mean, color='#41b6e6', label='mean', linewidth=3)
       my_plt.axvline(mode, color='#96ceb4', label='moda', linewidth=3)

       my_plt.xlabel(title)
       my_plt.title("Number of people have " + title)
       my_plt.legend()
       my_plt.show()

   def draw_boxplot(my_plt, my_data, title):
       x_axis = my_data.loc[:, title]
       my_plt.boxplot(x_axis, vert=False, showmeans=True, meanline=True)

       my_plt.title("Box plot for " + title)
       my_plt.xlabel(title)
       my_plt.show()

   def draw_scatter_plot(my_data, my_plt, x_axis, y_axis):
       my_plt.scatter(my_data[x_axis], my_data[y_axis])
       my_plt.xlabel(x_axis)
       my_plt.ylabel(y_axis)
       my_plt.title("Scatter plot to show relation between " + x_axis + " and " + y_axis)
       my_plt.show()

   def calculate_frequency_and_draw(my_data, my_plt, x_label):

       column_data = []  # to add data
       x_axis = []  # to set data for x-axis
       y_axis = []  # to set data for y-axis

       for i in my_data[x_label]:
           column_data.append(i)

       main_data = collections.Counter(column_data)  # to count how many times number appear
       for value, frequency in main_data.items():
           x_axis.append(value)
           y_axis.append(frequency)
       my_data.draw_bar_chart(my_plt, x_axis, y_axis, x_label)
       my_data.draw_pie_chart(my_plt, x_axis, y_axis, x_label)

   def calculate_IQR(data):
       q3, q1 = np.percentile(data, [75, 25])
       IQR = q3 - q1
       return IQR

   def calculate_central_tendency_and_dispersion(all_data, data_col):
       my_central_tendency = all_data.loc[:, data_col]

       mean = my_central_tendency.mean()
       median = my_central_tendency.median()
       mode = stat.multimode(my_central_tendency)

       print("********** Central Tendency For " + data_col + " **********\n")
       print("The mean of " + data_col + " is " + str(mean))
       print("The median of " + data_col + " is " + str(median))
       print("The mode of " + data_col + " is " + str(mode) + "\n")

       if len(mode) == 2:
           print("The distribution shape is BIMODAL")
       else:
           if mode[0] < median:
               print("The distribution shape is POSITIVE SKEWNESS")
           elif mode[0] > median:
               print("The distribution shape is NEGATIVE SKEWNESS")
           else:
               print("The distribution shape is NORMAL DISTRIBUTION")

               print("\n************************************************************************\n")
       print("********** Dispersion For " + data_col + " **********\n")
       print("The variance of " + data_col + " is " + str(my_central_tendency.var()))
       print("The standard deviation of " + data_col + " is " + str(my_central_tendency.std()))
       print("The Range of data : ")
       print("\nMinimum Value = " + str(min(my_central_tendency)))
       print("Maximum Value = " + str(max(my_central_tendency)))
       print("\nThe IQR of " + data_col + " is " + str(all_data.calculate_IQR(my_central_tendency)))

   def box_plot(self):
       data_Age =self.data_frame.loc[:, 'Age']
       box = plt.boxplot(data_Age, vert=0, patch_artist=True)
       colors = ['cyan', 'lightblue', 'lightgreen', 'tan']
       for patch, color in zip(box['boxes'], colors):
           patch.set_facecolor(color)
       plt.title('Average of Ages')
       plt.xlabel('Ages')
       plt.show()

   def his_for_Age(self):
       sns.set_style('whitegrid')
       sns.distplot(self.data_Age, fit=norm)  # "fit = norm"=> shows normal distribution on the graph
       plt.show()

   def percentage(self):
       F = self.data_frame.loc[self.data_frame['Sex'] == 1].count()[0]
       M = self.data_frame.loc[self.data_frame['Sex'] == 0].count()[0]
       labels = ['Female', 'Male ']
       explode = [0, 0.05]
       colors = ['pink', 'lightblue']
       plt.pie([M, F], labels=labels, autopct='%.2f %%', colors=colors, explode=explode)
       plt.title('Percentage of Male & Female ')
       plt.show()

   # Histogram for RestingBP and get the median for it
   def newResting(self):
       RestingBP_data = self.data_frame.loc[:, 'RestingBP']
       bins = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
       plt.hist(self.data_frame.RestingBP, bins=bins, edgecolor='black')
       plt.xticks(bins)
       plt.axvline(np.median(RestingBP_data), color='red', label='RestingBP Median')
       plt.axvline(np.mean(RestingBP_data), color='green', label='RestingBP mean and mode')
       plt.xlabel('RestingBP')
       plt.ylabel(' Frequency')
       plt.title('Average of RestingBP')
       plt.legend()
       print('the shape is almost Normal Distribution ', '\n')
       plt.show()

   # Kernal density estimation for"estimate probability density function from RestingBP"

   def density_function_from_RestingBP(self):
       sns.kdeplot(self.data_frame['RestingBP'], shade=True, cumulative=True)
       plt.title("Probability cumulative function for RestingBP")
       plt.show()

   # Plotting Max heart rate
   def Max_heart_rate(self):
       plt.hist(self.data_frame['MaxHR'])
       plt.bar(self.data_frame['MaxHR'].mean(), height=193, color='g', label='mode')
       plt.bar(self.data_frame['MaxHR'].mode(), height=193, color='r', label='Mean')
       plt.bar(self.data_frame['MaxHR'].median(), height=193, color='y', label='median')
       plt.legend()
       plt.xlabel('value', size=22)
       plt.ylabel('frequency', size=22)
       plt.title('Max Heart Rate', size=25)
       plt.show
       print('Skewness >>  ')
       print(self.data_frame['MaxHR'].skew())

   # ploting max hr as a box plot
   def max_hr_as_a_box_plot(self):

       plt.figure(figsize=(18, 10))
       plt.subplot(2, 2, 1)
       plt.title('MaxHR')
       plt.boxplot(self.data_frame['MaxHR'], vert=False)
       plt.bar(self.data_frame['MaxHR'].mean(), height=0.93, color='g', label='mode')
       plt.bar(self.data_frame['MaxHR'].mode(), height=0.93, color='r', label='Mean')
       plt.bar(self.data_frame['MaxHR'].median(), height=0.93, color='y', label='median')
       plt.legend()
       plt.show()

   ##Plotting Exercise Angina
   def Exercise_Angina(self):
      self.data_frame.groupby('ExerciseAngina').size().plot(kind='pie',  autopct='%.2f',textprops={'fontsize': 20},labels=['No', 'Y es'],legend=True).set_xlabel('Excercise Angina', size=22)

   # Plotting Heart Disease
   def Heart_Disease(self):
        self.data_frame.groupby('HeartDisease').size().plot(kind='pie',
                                                       autopct='%.2f',
                                                       textprops={'fontsize': 20},
                                                       label='',
                                                       labels=['No', 'Y es'],
                                                       legend=True).set_xlabel('Heart Diseases', size=22)

   # Plotting Resting ElectroCardioGraph
   def Resting_ElectroCardioGraph(self):
       self.data_frame.groupby('RestingECG').size().plot(kind='pie', textprops={'fontsize': 20},
                                                    colors=['gold', 'skyblue', 'red'], ylabel='').set_xlabel(
           'Resting ElectroCardioGraph', size=22)
       plt.show()

   # Plotting Resting ElectroCardioGraph
   def Resting_ElectroCardioGraph2(self):
     self.data_frame.groupby('ST_Slope').size().plot(kind='pie', textprops={'fontsize': 20},
                                                  colors=['gold', 'silver', 'pink'],
                                                  ylabel='').set_xlabel('ST_slope', size=22)
     plt.show()

   # Plotting Oldpeak ~Samy's code 'Porfecto'
   def Oldpeak(self):
      self.data_frame.Oldpeak.hist()
      print('\t\t\t\t\t\tOldpeak')
      plt.show()

   # # Histogram Shape : Spiked
   #
   # def ChestPainType(self):
   #     self.calculate_frequency_and_draw(self.data_frame, plt, "ChestPainType")
   #
   #
   # def FastingBS(self):
   #     self.calculate_frequency_and_draw(self.data_frame, plt, "FastingBS")
   #
   #
   # def histoCholesterol(self):
   #    self.draw_histogram(plt, self.data_frame, "Cholesterol")
   #
   #
   # def boxCholesterol(self):
   #     self.draw_boxplot(plt, self.data_frame, "Cholesterol")


   # def central_tendency_and_dispersion(self):
   #   self.calculate_central_tendency_and_dispersion(self.data_frame, "Cholesterol")

   # Calculate Q1 &Q2 & Q3
   def calculate(self):
       Q1 = np.percentile(self.data_Age, 25, interpolation='midpoint')
       Q2 = np.percentile(self.data_Age, 50, interpolation='midpoint')
       Q3 = np.percentile(self.data_Age, 75, interpolation='midpoint')
       IQR = Q3 - Q1
       low_limit = Q1 - (1.5 * IQR)
       upper_limit = Q3 + (1.5 * IQR)
       outlier = []
       for i in self.data_Age:
           if ((i < low_limit) or (i > upper_limit)):
               outlier.append(i)
               print('Outlier = ', outlier)
       print('Q1 = ', Q1, '\n', 'Q2 = ', Q2, '\n', 'Q3 = ', Q3, '\n')
       print('Interquartile range  = ', IQR, ' \n', 'Range = [', low_limit, ',', upper_limit, ']', '\n')
       if (Q3 - Q2 == Q2 - Q1):
           print('the shape is normal distribution', '\n')
       if (Q3 - Q2 < Q2 - Q1):
           print('the shape skweed to the left', '\n')
       if (Q3 - Q2 > Q2 - Q1):
           print('the shape skweed to the right', '\n')

   def ploting_modified_dataset(self):  # Ploting the whole modified dataset ~My code
       self.modified_x.hist()
       plt.show()

   def calculate_confidence_interval_critical_z(column_name, data, sample_size):
       population_data = np.array(data.loc[:, column_name])

       sample_of_column = np.random.choice(a=population_data, size=sample_size)
       sample_mean = sample_of_column.mean()
       # confidence interval of 95%
       z = stats.norm.ppf(q=0.975)
       # print(z_critical)

       population_std = population_data.std()

       margin_of_error = z * (population_std / math.sqrt(sample_size))

       confidence_interval = [(sample_mean - margin_of_error), (sample_mean + margin_of_error)]

       print("The confidence interval using critical-z for " + column_name + " = " + str(confidence_interval))

       # to draw the error bar
       intervals = []
       sample_means = []

       for sample_of_column in range(100):  # create 100 randome sample and draw them
           sample_of_column = np.random.choice(a=population_data, size=sample_size)

           sample_mean = sample_of_column.mean()
           sample_means.append(sample_mean)

           z = stats.norm.ppf(q=0.975)
           population_std = population_data.std()

           margin_of_error = z * (population_std / math.sqrt(sample_size))

           confidence_interval = [(sample_mean - margin_of_error), (sample_mean + margin_of_error)]

           intervals.append(confidence_interval)

       # plt.figure(figsize=(20, 20))
       plt.errorbar(x=np.arange(0.1, 100, 1),  # np.arange(start=0.1, stop=25, step=1)
                    y=sample_means,
                    yerr=[(top - bot) / 2 for top, bot in intervals],
                    fmt='o')
       plt.hlines(xmin=0, xmax=240, y=sample_mean, linewidth=2, color='red')
       plt.title("Error bar for " + column_name)
       plt.show()

   # def confidence_interval_critical_z(self):
   #     self.calculate_confidence_interval_critical_z("Cholesterol", self.data_frame, 7)

   def calculate_proportion_estimate_for_categorical_data(column_name, data, sample_size):
       population_data = np.array(data.loc[:, column_name])
       sample_of_column = random.sample(list(population_data), sample_size)

       if (column_name == "FastingBS"):
           for x in set(sample_of_column):
               if (x == '1'):
                   print("Having fasting blood sugar proportion estimate = " + str(
                       (sample_of_column.count(x) / sample_size) * 100) + "%")
               else:
                   print("Haven't fasting blood sugar proportion estimate = " + str(
                       (sample_of_column.count(x) / sample_size) * 100) + "%")

       elif (column_name == "HeartDisease"):
           for x in set(sample_of_column):
               if (x == '1'):
                   print("Having heart disease proportion estimate = " + str(
                       (sample_of_column.count(x) / sample_size) * 100) + "%")
               else:
                   print("Haven't heart disease proportion estimate = " + str(
                       (sample_of_column.count(x) / sample_size) * 100) + "%")

       elif (column_name == "ExerciseAngina"):
           for x in set(sample_of_column):
               if (x == '1'):
                   print("Doing exercise angina proportion estimate = " + str(
                       (sample_of_column.count(x) / sample_size) * 100) + "%")
               else:
                   print("don't make exercise angina proportion estimate = " + str(
                       (sample_of_column.count(x) / sample_size) * 100) + "%")
       else:
           for x in set(sample_of_column):
               print(x + " proportion estimate = " + str((sample_of_column.count(x) / sample_size) * 100) + "%")

   # def calculate_proportion_estimate_forcategorical_data(self):
   #     self.calculate_proportion_estimate_for_categorical_data("Cholesterol", self.data_frame, 2)

   # to get T_test for the age ,T_test is t_distribution  equal to ((xBar - mean of population) /(Standard dviation for smple /root n))
   # we need to get mean of population from sample mean but we don't know the population SD for it ,so thet we use the Hypothis test
   def get_T_test_for_age(self,loc_column):
       data_age = self.data_frame.loc[:,loc_column]
       ages_mean = np.mean(data_age)

       ttest, p_value = ttest_1samp(data_age, 60)
       if p_value < 0.05:
           print("we are rejecting null hypothesis ", '\n')
       else:
           print("we are accepted null hypothesis", '\n')

   def draw_correlations(self):
       correlations = pd.get_dummies(self.data_frame)
       corrmat = correlations.corr()

       cg = sns.clustermap(corrmat, cmap="YlGnBu", linewidths=0.1);
       plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
       plt.show()
   def calculate_correlations(self):
       correlations = pd.get_dummies(self.data_frame)
       correlations.corr()



