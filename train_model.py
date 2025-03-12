import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import re

df = pd.read_excel(r"C:\Users\Dell\Desktop\company work\MarksRange-ExpectedRank2024-Corrected.xlsx", engine="openpyxl")


def extract_min_marks(value):
    # Ensure the value is a string
    if not isinstance(value, str):
        return np.nan  # Return NaN for non-string values
    
    # Extract numbers using regex
    numbers = re.findall(r'\d+', value)  
    
    if not numbers:
        return np.nan  # Return NaN if no numbers are found
    
    return int(numbers[0])  # Convert first number found to integer

df['Marks'] = df['Marks Range'].apply(extract_min_marks)

df['Rank'] = df['Expected Rank 2024'].apply(
    lambda x: int(re.sub(r'[^0-9]', '', x.split('–')[0])) if isinstance(x, str) and re.search(r'\d', x) else np.nan
)

df['Difficulty'] = 'medium'

X = df[['Marks', 'Difficulty']]
y = df['Rank']

X['Difficulty'] = X['Difficulty'].map({'easy': 0, 'medium': 1, 'hard': 2})

scaler = MinMaxScaler()
X['Marks'] = scaler.fit_transform(X[['Marks']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Metrics:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("R² Score:", r2_score(y_test, y_pred_rf))


joblib.dump(rf_model, r"C:\Users\Dell\Desktop\company work\random_forest_model.pkl")

joblib.dump(scaler, r"C:\Users\Dell\Desktop\company work\scaler.pkl")