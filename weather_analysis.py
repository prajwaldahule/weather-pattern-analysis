# Weather Pattern Analysis
# internship project 4 - data analytics
# Prajwal Jitendra Dahule | CITS1288

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# load dataset
df = pd.read_csv('weather_data.csv')
print("data loaded")
print("shape:", df.shape)
print(df.head())


# ------- data cleaning -------

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')

print("\nchecking nulls:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
print("records after cleaning:", len(df))

# adding avg temperature column
df['Temp_Avg'] = (df['Temperature_Max'] + df['Temperature_Min']) / 2

print("\ncities in dataset:", df['City'].unique())
print("months covered:", df['Month_Name'].unique())


# ------- analysis -------

print("\n--- overall stats ---")
print(df[['Temperature_Max', 'Temperature_Min', 'Humidity', 'Rainfall_mm', 'Wind_Speed_kmh']].describe())

# city wise average temperature
city_temp = df.groupby('City')['Temp_Avg'].mean().sort_values(ascending=False)
print("\naverage temperature by city:")
print(city_temp)

# monthly average temperature (all cities combined)
monthly_temp = df.groupby(['Month', 'Month_Name'])['Temp_Avg'].mean().reset_index()
monthly_temp = monthly_temp.sort_values('Month')

# city wise total rainfall
city_rain = df.groupby('City')['Rainfall_mm'].sum().sort_values(ascending=False)
print("\ntotal rainfall by city:")
print(city_rain)

# monthly rainfall
monthly_rain = df.groupby(['Month', 'Month_Name'])['Rainfall_mm'].sum().reset_index()
monthly_rain = monthly_rain.sort_values('Month')

# weather condition counts
condition_counts = df['Weather_Condition'].value_counts()
print("\nweather conditions frequency:")
print(condition_counts)


# ------- visualizations -------

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Weather Pattern Analysis - 4 Cities (Jan-Jun 2024)', fontsize=17, fontweight='bold')

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
city_colors = {'Mumbai': '#4472C4', 'Delhi': '#ED7D31',
               'Bangalore': '#A9D18E', 'Chennai': '#FF6B6B'}
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai']

# chart 1 - temperature trend per city
ax1 = axes[0, 0]
for city in cities:
    city_data = df[df['City'] == city].groupby('Month')['Temp_Avg'].mean()
    ax1.plot(month_labels, city_data.values, marker='o',
             label=city, linewidth=2, color=city_colors[city])
ax1.set_title('Monthly Avg Temperature by City')
ax1.set_ylabel('Temperature (°C)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.4)

# chart 2 - monthly rainfall bar chart
ax2 = axes[0, 1]
bars = ax2.bar(month_labels, monthly_rain['Rainfall_mm'].values,
               color=['#4472C4', '#4472C4', '#A9D18E', '#ED7D31', '#FF6B6B', '#FF6B6B'])
ax2.set_title('Monthly Total Rainfall (All Cities)')
ax2.set_ylabel('Rainfall (mm)')
for bar, val in zip(bars, monthly_rain['Rainfall_mm'].values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}mm', ha='center', fontsize=9)
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

# chart 3 - city wise rainfall comparison
ax3 = axes[1, 0]
rain_colors = [city_colors[c] for c in city_rain.index]
bars2 = ax3.bar(city_rain.index, city_rain.values, color=rain_colors)
ax3.set_title('Total Rainfall by City (6 Months)')
ax3.set_ylabel('Total Rainfall (mm)')
for bar, val in zip(bars2, city_rain.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val}mm', ha='center', fontsize=10, fontweight='bold')
ax3.grid(True, axis='y', linestyle='--', alpha=0.4)

# chart 4 - humidity heatmap city vs month
ax4 = axes[1, 1]
pivot_humidity = df.pivot_table(values='Humidity', index='City',
                                columns='Month_Name', aggfunc='mean')
ordered = [m for m in ['January','February','March','April','May','June'] if m in pivot_humidity.columns]
pivot_humidity = pivot_humidity[ordered]
sns.heatmap(pivot_humidity, ax=ax4, annot=True, fmt='.0f',
            cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Humidity %'})
ax4.set_title('Average Humidity - City vs Month')
ax4.tick_params(axis='x', rotation=30)

# chart 5 - weather condition distribution pie
ax5 = axes[2, 0]
top_conditions = condition_counts.head(6)
colors_pie = ['#4472C4', '#ED7D31', '#A9D18E', '#FF6B6B', '#9C27B0', '#00BCD4']
ax5.pie(top_conditions.values, labels=top_conditions.index,
        autopct='%1.1f%%', colors=colors_pie,
        startangle=90, wedgeprops={'edgecolor': 'white'})
ax5.set_title('Weather Condition Distribution')

# chart 6 - max temperature comparison all cities monthly
ax6 = axes[2, 1]
width = 0.2
x = np.arange(len(month_labels))
for i, city in enumerate(cities):
    city_data = df[df['City'] == city].groupby('Month')['Temperature_Max'].mean()
    ax6.bar(x + i * width, city_data.values, width=width,
            label=city, color=city_colors[city], alpha=0.85)
ax6.set_title('Monthly Max Temperature by City')
ax6.set_ylabel('Max Temperature (°C)')
ax6.set_xticks(x + width * 1.5)
ax6.set_xticklabels(month_labels)
ax6.legend()
ax6.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('weather_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()
print("charts saved")


# ------- prediction model -------

# predicting next month avg temperature for Mumbai
mumbai_df = df[df['City'] == 'Mumbai'].groupby('Month')['Temp_Avg'].mean().reset_index()
X = mumbai_df['Month'].values.reshape(-1, 1)
y = mumbai_df['Temp_Avg'].values

model = LinearRegression()
model.fit(X, y)

july_pred = model.predict([[7]])[0]
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print(f"\nprediction model (Mumbai temperature):")
print(f"R2 score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"predicted avg temperature for Mumbai July 2024: {july_pred:.1f}°C")

# prediction chart
fig2, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X, y, color='#4472C4', s=80, zorder=5, label='Actual Temp (Mumbai)')
ax.plot(X, y_pred, color='#ED7D31', linewidth=2, label='Regression Line')
ax.scatter([7], [july_pred], color='green', s=150, zorder=6, marker='*',
           label=f'July Prediction: {july_pred:.1f}°C')
ax.set_title('Mumbai Temperature Prediction - July 2024')
ax.set_xlabel('Month')
ax.set_ylabel('Avg Temperature (°C)')
ax.set_xticks(range(1, 8))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul(pred)'])
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('weather_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
print("prediction chart saved")


# final summary
print("\n--- Final Summary ---")
print(f"total records: {len(df)}")
print(f"cities covered: {', '.join(cities)}")
print(f"hottest city: {city_temp.idxmax()} ({city_temp.max():.1f}°C avg)")
print(f"highest rainfall: {city_rain.idxmax()} ({city_rain.max()}mm)")
print(f"most common weather: {condition_counts.idxmax()}")
print(f"mumbai july prediction: {july_pred:.1f}°C")
