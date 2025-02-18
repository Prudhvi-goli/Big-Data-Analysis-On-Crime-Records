# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data Import and Pre-processing
# Load the dataset
df = pd.read_csv(CAD_2014_to_2017.csv)

# Print the actual column names in the DataFrame
print(df.columns)

# List of relevant columns for the model
columns_of_interest = ['year', 'month', 'day', 'dayofyear', 'dayofweek', 'hour','CDA', 'Division_', 'HoodId', 'premise_type']

# Filter the DataFrame to include only relevant columns
df_filtered = df[columns_of_interest]

# Factorize categorical variables
MCI_encoded = pd.factorize(df_filtered['CDA'])
df_filtered['CDA'] = MCI_encoded[0]
MCI_definitions = MCI_encoded[1]

premise_encoded = pd.factorize(df_filtered['premise_type'])
df_filtered['premise_type'] = premise_encoded[0]
premise_definitions = premise_encoded[1]

year_encoded = pd.factorize(df_filtered['year'])
df_filtered['year'] = year_encoded[0]
year_definitions = year_encoded[1]

month_encoded = pd.factorize(df_filtered['month'])
df_filtered['month'] = month_encoded[0]
month_definitions = month_encoded[1]

hood_encoded = pd.factorize(df_filtered['HoodId'])
df_filtered['HoodId'] = hood_encoded[0]
hood_definitions = hood_encoded[1]

hour_encoded = pd.factorize(df_filtered['hour'])
df_filtered['hour'] = hour_encoded[0]
hour_definitions = hour_encoded[1]

day_week_encoded = pd.factorize(df_filtered['dayofweek'])
df_filtered['dayofweek'] = day_week_encoded[0]
day_week_definitions = day_week_encoded[1]

day_year_encoded = pd.factorize(df_filtered['dayofyear'])
df_filtered['dayofyear'] = day_year_encoded[0]
day_year_definitions = day_year_encoded[1]

# One-Hot Encode 'Division' column
division_encoded = pd.get_dummies(df_filtered['Division_'], prefix='Division')
df_filtered = df_filtered.drop('Division_', axis=1)
df_filtered = pd.concat([df_filtered, division_encoded], axis=1)

# Prepare features and target variable
X = df_filtered.drop(['CDA'], axis=1).values
y = df_filtered['CDA'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False)
encoded_X = encoder.fit_transform(X)

# Split the encoded data into training and testing sets
X_train_OH, X_test_OH, y_train_OH, y_test_OH = train_test_split(encoded_X, y, test_size=0.25, random_state=21)

# Numeric Encoded Model with Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf, target_names=MCI_definitions))

# One-Hot Encoded Model with Random Forest
rf_classifier_OH = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
rf_classifier_OH.fit(X_train_OH, y_train_OH)
y_pred_rf_OH = rf_classifier_OH.predict(X_test_OH)

# Evaluate the One-Hot Encoded Random Forest model
print("One-Hot Encoded Random Forest Model Accuracy:", accuracy_score(y_test_OH, y_pred_rf_OH))
print("Confusion Matrix:\n", confusion_matrix(y_test_OH, y_pred_rf_OH))
print("Classification Report:\n", classification_report(y_test_OH, y_pred_rf_OH, target_names=MCI_definitions))

# Random Forest with Balanced Class Weight
rf_classifier_balanced = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, class_weight='balanced')
rf_classifier_balanced.fit(X_train, y_train)
y_pred_balanced = rf_classifier_balanced.predict(X_test)

# Evaluate the balanced Random Forest model
print("Balanced Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_balanced))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_balanced))
print("Classification Report:\n", classification_report(y_test, y_pred_balanced, target_names=MCI_definitions))

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, random_state=42)
gb_classifier.fit(X_train_OH, y_train_OH)  # Fit the model on the one-hot encoded training data

# Make predictions on the one-hot encoded test data
y_pred_gb_OH = gb_classifier.predict(X_test_OH)

# Evaluate the Gradient Boosting model
print("Gradient Boosting Model Accuracy:", accuracy_score(y_test_OH, y_pred_gb_OH))
print("Confusion Matrix:\n", confusion_matrix(y_test_OH, y_pred_gb_OH))
print("Classification Report:\n", classification_report(y_test_OH, y_pred_gb_OH, target_names=MCI_definitions))

# Summary of results
print("Summary of Model Accuracies:")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"One-Hot Encoded Random Forest Accuracy: {accuracy_score(y_test_OH, y_pred_rf_OH)}")
print(f"Balanced Random Forest Accuracy: {accuracy_score(y_test, y_pred_balanced)}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test_OH, y_pred_gb_OH)}")
