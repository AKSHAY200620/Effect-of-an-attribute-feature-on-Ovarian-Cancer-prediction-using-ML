import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


data = pd.read_excel(r'C:\Users\aksha\Downloads\Supplementary data 1.xlsx')
print(data.head())
print(data.columns)

# to standardize column names
data.columns = data.columns.str.strip()

# Check actual target column name
print(data.columns)
print(data['TYPE'].unique())

# Fill missing numeric values
data = data.fillna(data.mean(numeric_only=True))

# Separate features and target
X = data.drop('TYPE', axis=1)
X = X.select_dtypes(include='number')  # ensures only numeric
y = data['TYPE']

# Encode target if it's categorical
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

