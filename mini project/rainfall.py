import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('weather_test_data.csv')

# Replace categorical values for RainToday and RainTomorrow with numerical values
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

# Function to predict the possibility of rain tomorrow
def possible_rain():
    fig = plt.figure(figsize=(8, 5))
    df.RainTomorrow.value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'navy'], alpha=0.9, rot=0)
    plt.title('RainTomorrow Indicator: No(0) and Yes(1) in the Imbalanced Dataset')
    plt.show()
# Function to analyze rainfall data in India
def analyze():
    data = pd.read_csv("Weather Data in India from 1901 to 2017.csv")
    ax = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean().plot.bar(
        width=0.5, linewidth=2, figsize=(16, 10)
    )
    plt.xlabel('Month', fontsize=30)
    plt.ylabel('Monthly Rainfall (in mm)', fontsize=30)
    plt.title('Average Monthly Rainfall in India', fontsize=25)
    ax.tick_params(labelsize=10)
    plt.grid()
    plt.show()
