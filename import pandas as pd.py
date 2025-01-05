import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics

df = pd.read_csv('Airdata.csv')

df.head()

#Question 1
df = df.drop(columns=["Generosity", "Dystopia"])
df.head()

#Question 2
# Save the updated file 
df.to_csv("CS_2019_036_happy.csv", index=False)

#Question 3 
top_10_happiest = df.nlargest(10, 'Happiness Score')
print(top_10_happiest[['Country', 'Happiness Score']])

#Question 4 
plt.figure(figsize=(8, 6))
df['Economy'].hist(bins=15, color='skyblue', edgecolor='black')
plt.title('Histogram of Economy')
plt.xlabel('Economy')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()
# Calculating Skewness for Columns
def interpret_skewness(skewness):
 if skewness < 0:
 return "Negatively Skewed"
 elif skewness > 0:
 return "Positively Skewed"
 else:
 return "Symmetric"
economy_skewness = df['Economy'].skew()
print("Economy Skewness:", economy_skewness, "-", interpret_skewness(economy_skewness))

#Question 5
# Histogram for Health
plt.figure(figsize=(8, 6))
df['Health'].hist(bins=15, color='lightgreen', edgecolor='black')
plt.title('Histogram of Health')
plt.xlabel('Health')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()
# Calculating Skewness for Columns
def interpret_skewness(skewness):
 if skewness < 0:
 return "Negatively Skewed"
 elif skewness > 0:
 return "Positively Skewed"
 else:
 return "Symmetric"
health_skewness = df['Health'].skew()
print("Health Skewness:", health_skewness, "-", interpret_skewness(health_skewness))

#Question 6
# Pie Chart for Region-wise Job Satisfaction
region_job_satisfaction = df.groupby('Region')['Job Satisfaction'].mean()
plt.figure(figsize=(8, 8))
region_job_satisfaction.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Region-wise Job Satisfaction')
plt.ylabel('')
plt.show()

#Question 8
# Happiness Score
happiness_mean = df['Happiness Score'].mean()
happiness_median = df['Happiness Score'].median()
print("Happiness Score - Mean:", happiness_mean, "Median:", happiness_median)

#Question 10
# Freedom Score
freedom_mean = df['Freedom'].mean()
freedom_median = df['Freedom'].median()
print("Mean of Freedom Score:", freedom_mean, "Median of Freedom Score:", freedom_median)

#Question 12 
# Happiness Score
happiness_quartiles = {
 "Q1": df['Happiness Score'].quantile(0.25),
 "Q2": df['Happiness Score'].quantile(0.5),
 "Q3": df['Happiness Score'].quantile(0.75)}
print("Happiness Score - Quartiles:")
for key, value in happiness_quartiles.items():
 print(f"{key}: {value}")

#Question 13 
def quantile_skewness(q1, q2, q3):
 left_length = q2 - q1
 right_length = q3 - q2
 if right_length > left_length:
 return "Positive Quantile Skewness"
 elif left_length > right_length:
return "Negative Quantile Skewness"
 else:
 return "No Quantile Skewness"
# Calculate and print skewness based on quantiles for Happiness Score
q1 = happiness_quartiles['Q1']
q2 = happiness_quartiles['Q2']
q3 = happiness_quartiles['Q3']
print("Quantile Skewness of Happiness Score:", quantile_skewness(q1, q2, q3))

#Question 14
# Freedom Score
freedom_quartiles = {
 "Q1": df['Freedom'].quantile(0.25),
 "Q2": df['Freedom'].quantile(0.5),
 "Q3": df['Freedom'].quantile(0.75)}
print("Freedom Score - Quartiles:")
for key, value in freedom_quartiles.items():
 print(f"{key}: {value}")

#Question 15 
def quantile_skewness(q1, q2, q3):
 left_length = q2 - q1
 right_length = q3 - q2
 if right_length > left_length:
 return "Positive Quantile Skewness"
 elif left_length > right_length:
 return "Negative Quantile Skewness"
 else:
 return "No Quantile Skewness"
# Calculate and print skewness based on quantiles for Freedom Score
q1_freedom = freedom_quartiles['Q1']
q2_freedom = freedom_quartiles['Q2']
q3_freedom = freedom_quartiles['Q3']
print("Quantile Skewness of Freedom Score:", quantile_skewness(q1_freedom, q2_freedom, 
q3_freedom))

#Question 16
# Maximum and Minimum for Happiness Score
happiness_max = df['Happiness Score'].max()
happiness_min = df['Happiness Score'].min()
# Print the results
print("Happiness Score - Maximum:", happiness_max, "Minimum:", happiness_min)

#Question 17
# Maximum and Minimum for Freedom Score
freedom_max = df['Freedom'].max()
freedom_min = df['Freedom'].min()
# Print the results
print("Freedom Score - Maximum:", freedom_max, "Minimum:", freedom_min)

#Question 18
happiness_range = df['Happiness Score'].max() - df['Happiness Score'].min()
happiness_iqr = df['Happiness Score'].quantile(0.75) - df['Happiness Score'].quantile(0.25)
happiness_variance = df['Happiness Score'].var()
happiness_std = df['Happiness Score'].std()
print("Range:", happiness_range)
print("IQR:", happiness_iqr)
print("Variance:", happiness_variance)
print("Std Dev:", happiness_std)

#Question 19
Freedom_range = df['Freedom'].max() - df['Freedom'].min()
Freedom_iqr = df['Freedom'].quantile(0.75) - df['Freedom'].quantile(0.25)
Freedom_variance = df['Freedom'].var()
Freedom_std = df['Freedom'].std()
print("Range:", Freedom_range)
print("IQR:", Freedom_iqr)
print("Variance:",Freedom_variance)
print("Std Dev:", Freedom_std)

#Question 20 
Q1 = df['Economy'].quantile(0.25)
Q3 = df['Economy'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Economy'] < lower_bound) | (df['Economy'] > upper_bound)]
print("Outliers:", outliers)

#Question 21 
print("Five-number summary for Economy:")
print(df['Economy'].describe())
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Economy'], color='Yellow')
plt.title('Box Plot of Economy')
plt.xlabel('Economy')
plt.show()

#Question 22
cov_health_happiness = df['Happiness Score'].cov(df['Health'])
corr_health_happiness = df['Happiness Score'].corr(df['Health'])
print("Covariance:", cov_health_happiness, "Correlation:", corr_health_happiness)

#Question 23 
cov_Corruption_happiness = df['Happiness Score'].cov(df['Corruption'])
corr_Corruption_happiness = df['Happiness Score'].corr(df['Corruption'])
print("Covariance:", cov_Corruption_happiness, "Correlation:", corr_Corruption_happiness)

#Question 24
# Scatter Plots
plt.figure(figsize=(8, 6))
plt.scatter(df['Health'], df['Happiness Score'], color='blue')
plt.title('Health vs Happiness Score')
plt.xlabel('Health')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.show()



