# Heart diseases prediction using Machine learning


## Machine Learning approach:
-	To make good use of our dataset we decided to train 3 different machine learning models such that each model depends on different algorithms than the others.
-	Before being able to train the models, dataset had to be prepared first by removing duplicates, null or missing values, outliers and using dummies for categorial data (like chestPainType)
-	All models were trained using a part of the data set and tested using the other part
<br>	
--	The first model was trained using multiple linear regression, it had poor accuracy of only 55%
<br>	

--	Second model was trained using decision tree, and it was a great leap forward as the accuracy has improved to about 80% to 85%, depending on the random portion of dataset used during the training.
<br>	
--	Third model was trained using K-Nearest-Neighbors (KNN) algorithm and had accuracy of 85%
<br>	
-	“scikit-learn” library from “Conda” environment with python 3.9 was used to train the models and “joblib” to save them as binaries for later use
-	 A Jupyter notebook was used to divide code into cells to be easy for reading and documenting along the whole study 
-	To test for an observation, the required attributes need to be passed in form of (.CSV) file to the model
-	Each model produces output of:
either (1: “have heart disease”) or (0: “Doesn’t have heart disease”)
 
## Introduction:
	
  This project aims to study the likelihood of having heart diseases,
Which is a quite interesting subject to study where:
-	Heart disease is the leading cause of death in the United States and globally.
-	One person dies every 36 seconds in the United States from cardiovascular disease.
-	About 659,000 people in the United States die from heart disease each year, that’s 1 in every 4 deaths.
Source: www.cdc.gov/heartdisease
<br>
So, to help People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia ‘excessive fats’ or already established disease) who need early detection and management. <br>

I decided to analyze this subject statistically using graphical and numerical methods and train a machine learning model that can make good use of the dataset.
<br>
## Summary of research:
### Dataset:
The dataset is based on a combination of 5 datasets with 11 common features, it contains 918 observations after removing duplicates
Used datasets are:
-	Cleveland: 303 observations
-	Hungarian: 294 observations
-	Switzerland: 123 observations
-	Long Beach VA: 200 observations
-	Stalog (Heart) Data Set: 270 observations
<br>

### Dataset source: 
fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [December 2021] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.
<br>

### Dataset details:
the dataset comes in form of a ‘‘.CSV’’ file, it will be analyzed using “Pandas” and “Matplotlib” libraries on python 3.9 using “Jupyter Notebooks” with “Conda” environment and it contains 12 attributes divided into columns which are: <br>

-	Age: age of the patient [years]
-	Sex: sex of the patient [M: Male, F: Female]
-	ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
-	RestingBP: resting blood pressure [mm Hg]
-	Cholesterol: serum cholesterol [mm/dl]
-	FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
-	RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
-	MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
-	ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
-	Oldpeak: oldpeak = ST [Numeric value measured in depression]
-	ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
-	HeartDisease: output class [1: heart disease, 0: Normal]
<br>

## Exploratory data Analysis:
By studying and analyzing this data, The following results were deduced:
 
### Age:
-	Mean of Ages: 53.510893246187365 
-	Median of Ages: 54.0 
-	Mode of Ages: 54 repeated 51 times
-	Variance of Ages: 88.9742 
-	Standard deviation of Ages: 9.4326
-	Q1= 47.0, Q2 = 54.0, Q3 = 60.0 
-	Interquartile range (IQR) =1 3.0  
-	Range: minimum: 27.5, maximum 79.5
-	The distribution is normal and slightly skewed to the left 
### Sex:
-	Males are: 78.98% of the sample, while
-	Females are: 21.01% of it.

### Chest pain type:
-	54% of observations had Asymptomatic pain (ASY)
-	22.1% had Non-Anginal Pain (NAP)
-	18.8% had Atypical Angina (ATA)
-	5% Typical Angina (TA)
<br>

### RestingBP:
-	Mean of RestingBP: 132.39651416122004 
-	Median of RestingBP: 130.0 
-	Mode of RestingBP: 120 repeated 132 times
-	Variance of RestingBP: 342.7739 
-	Standard deviation of RestingBP: 18.5141

<br>

### Cholesterol:
-	Mean of Cholesterol is 244.62875816993574
-	Median of Cholesterol is 244.6
-	Mode of Cholesterol is 244.6
-	Distribution of shape is NORMAL DISTRIBUTION
-	Variance of Cholesterol is 2842.8124
-	Standard deviation of Cholesterol is 53.3180
-	Range of data:
-	Minimum Value = 85.0
-	Maximum Value = 603.0
-	IQR of Cholesterol is 53.0
<br>

### FastingBs:
-	76.7% of observations had fasting blood sugar greater than 120 mg/dl
-	23.3% of observations had fasting blood sugar less than 120 mg/dl
<br>

### RestingECG:
-	60.13% of observation are Normal
-	19.39% of observation having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
-	20.48% LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria
<br>

### MaxHR:
-	Mean: 136.8093
-	Median: 138
-	Mode: 150
-	Standard deviation: 25.4603
-	Variance: 648.2268
-	Q1: 120
-	Q2: 138
-	Q3: 156
-	IQR: 36
-	Minimum: 60
-	Maximum: 202
-	Range: 142
Distribution is normal, Skewness: -0.1443    
Slightly skewed to the left
<br>
### ExerciseAngina:
-	59.59% of observation didn’t have Exercise Angina
-	40.41% of observation had Exercise Angina
<br>

### Oldpeak:
-	Mean of Oldpeak is 0.8873
-	Median of Oldpeak is 0.6
-	Mode of Oldpeak is [0.0]
-	distribution shape is positively skewed
-	Variance of Oldpeak is 1.1375
-	Standard deviation of Oldpeak is 1.0665
-	Minimum Value = -2.6
-	Maximum Value = 6.2
-	IQR of Oldpeak is 1.5

<br>

### ST_Slope:
the slope of the peak exercise ST segment
-	50.11% have FLAT slope
-	43.03% have an UP slope
-	6.86% have a DOWN slope

<br>

### HeartDisease:
-	44.66% of the observations didn’t have heart disease
-	55.34% of the observations had heart disease

<br>

## Correlation of 2D data:

-	The confidence interval using critical-z for Cholesterol = [234.9915, 255.8804]
-	The confident interval using critical-t for Cholesterol = [235.6967, 256.9592]
-	The confidence interval using critical-z for Age = [50.3622, 54.0577] 
-	The confident interval using critical-t for Age = (51.6281, 55.4918) 
-	The confidence interval using critical-z for RestingBP = [131.6432, 138.8967] 
-	The confident interval using critical-t for RestingBP = (128.7058, 135.3941) 
-	The confidence interval using critical-z for MaxHR = [129.6925, 139.6674]
-	The confident interval using critical-t for MaxHR = (127.5521, 137.0278)
-	Haven't fasting blood sugar proportion estimate = 79.0%
-	Haven't fasting blood sugar proportion estimate = 21.0%
-	Haven't heart disease proportion estimate = 47.0%
-	Haven't heart disease proportion estimate = 53.0%
-	don't make exercise angina proportion estimate = 59.0%
-	don't make exercise angina proportion estimate = 41.0%

<br>


