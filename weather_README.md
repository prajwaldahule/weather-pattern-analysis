# Weather Pattern Analysis

**Intern ID: CITS1288**
**Name: Prajwal Jitendra Dahule**
**Organization: Codtech IT Solutions Pvt. Ltd**
**Internship Period: 20 May 2026 - 17 June 2026**

---

This is my fourth and last internship project. I analyzed weather data for 4 Indian cities — Mumbai, Delhi, Bangalore and Chennai — to find temperature trends, rainfall patterns and humidity changes over 6 months.

---

## Tools and Libraries Used

- Python 3
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset

I created a dummy dataset with 104 records covering January to June 2024 for 4 cities.

Columns:
- Date
- City (Mumbai, Delhi, Bangalore, Chennai)
- Temperature_Max (°C)
- Temperature_Min (°C)
- Humidity (%)
- Rainfall_mm
- Wind_Speed_kmh
- Weather_Condition
- UV_Index

---

## Steps I Followed

**1. Data Loading**
Loaded the CSV and checked shape and columns.

**2. Data Cleaning**
- Converted Date to datetime format
- Extracted month name
- Added Temp_Avg column (max + min / 2)
- Checked for null values — none found
- Removed duplicates

**3. Analysis**
- City wise average temperature and total rainfall
- Monthly temperature and rainfall trends
- Weather condition frequency
- Humidity patterns across months

**4. Visualization**
Made 6 charts:
1. Monthly avg temperature by city (line chart)
2. Monthly total rainfall (bar chart)
3. City wise total rainfall comparison (bar chart)
4. Humidity heatmap — city vs month
5. Weather condition distribution (pie chart)
6. Monthly max temperature by city (grouped bar chart)

**5. Prediction Model**
Used Linear Regression to predict Mumbai's average temperature for July 2024.

---

## Results

- Hottest city: Mumbai (30.7°C average)
- Highest rainfall: Mumbai (573mm in 6 months)
- Most common weather condition: Sunny
- Delhi had highest temperatures in May (41-42°C)
- Bangalore had most rainfall events (monsoon effect)
- Predicted Mumbai July 2024 temperature: 29.5°C

---

## Files in this Project

- weather_data.csv — dataset
- weather_analysis.py — main python code
- weather_visualizations.png — all 6 charts
- weather_prediction.png — prediction chart
- README.md — this file

---

## How to Run

```
pip install pandas matplotlib seaborn scikit-learn
python weather_analysis.py
```
