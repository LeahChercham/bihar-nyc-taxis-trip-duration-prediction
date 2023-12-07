from sklearn.model_selection import train_test_split
import pandas as pd
import commun

# Load your data from CSV
data = pd.read_csv(commun.DB_PATH)

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['trip_duration'])
y = data['trip_duration']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=commun.TEST_SIZE, random_state=commun.RANDOM_STATE)

# Save the training and testing sets to separate CSV files
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)
