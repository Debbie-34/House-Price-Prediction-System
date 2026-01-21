# House Price Prediction - Model Development
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
# Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
df = pd.read_csv('train.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# 2. Data Preprocessing

# Select the 6 features from the recommended 9
# Choosing: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood
selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 
                     'GarageCars', 'YearBuilt', 'Neighborhood']
target = 'SalePrice'

# Create working dataframe
df_work = df[selected_features + [target]].copy()

print("\nSelected features info:")
print(df_work.info())
print("\nMissing values:")
print(df_work.isnull().sum())

# a. Handle missing values
# Fill numerical missing values with median
numerical_features = ['TotalBsmtSF', 'GarageCars']
for col in numerical_features:
    if df_work[col].isnull().sum() > 0:
        df_work[col].fillna(df_work[col].median(), inplace=True)

# Fill categorical missing values with mode
if df_work['Neighborhood'].isnull().sum() > 0:
    df_work['Neighborhood'].fillna(df_work['Neighborhood'].mode()[0], inplace=True)

print("\nMissing values after handling:")
print(df_work.isnull().sum())

# c. Encode categorical variables
le = LabelEncoder()
df_work['Neighborhood_encoded'] = le.fit_transform(df_work['Neighborhood'])

# Save the label encoder for later use
joblib.dump(le, 'model/label_encoder.pkl')

# Prepare features and target
X = df_work[['OverallQual', 'GrLivArea', 'TotalBsmtSF', 
             'GarageCars', 'YearBuilt', 'Neighborhood_encoded']]
y = df_work[target]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

# d. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'model/scaler.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# 3 & 4. Implement and Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training completed!")

# 5. Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_pred_train)

# Testing metrics
test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)
print("\nTraining Set Performance:")
print(f"MAE:  ${train_mae:,.2f}")
print(f"MSE:  ${train_mse:,.2f}")
print(f"RMSE: ${train_rmse:,.2f}")
print(f"R²:   {train_r2:.4f}")

print("\nTesting Set Performance:")
print(f"MAE:  ${test_mae:,.2f}")
print(f"MSE:  ${test_mse:,.2f}")
print(f"RMSE: ${test_rmse:,.2f}")
print(f"R²:   {test_r2:.4f}")
print("="*50)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 
                'GarageCars', 'YearBuilt', 'Neighborhood'],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# 6. Save the trained model
joblib.dump(model, 'model/house_price_model.pkl')
print("\nModel saved successfully to 'model/house_price_model.pkl'")

# 7. Test reloading the model
print("\nTesting model reload...")
loaded_model = joblib.load('model/house_price_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')
loaded_le = joblib.load('model/label_encoder.pkl')

# Make a test prediction
test_input = X_test[0].reshape(1, -1)
original_pred = model.predict(test_input)[0]
reloaded_pred = loaded_model.predict(test_input)[0]

print(f"Original model prediction: ${original_pred:,.2f}")
print(f"Reloaded model prediction: ${reloaded_pred:,.2f}")
print(f"Predictions match: {np.isclose(original_pred, reloaded_pred)}")

print("\n✓ Model development completed successfully!")