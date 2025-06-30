import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
#import shap  # optional
#import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
data = pd.read_excel(r'C:\Users\aksha\Downloads\Supplementary data 1.xlsx')
data.columns = data.columns.str.strip()
data = data.fillna(data.mean(numeric_only=True))

# Step 2: Split features and target
X = data.drop('TYPE', axis=1).select_dtypes(include='number')
y = data['TYPE']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define individual models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True)
}

# Step 4: Add Stacking Classifier
stacking_estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svm', SVC(probability=True)),
    ('nv', GaussianNB()),
    ('xb', XGBClassifier(eval_metric='logloss'))
]
stack_model = StackingClassifier(estimators=stacking_estimators, final_estimator=RandomForestClassifier())
models["Stacking Classifier"] = stack_model

# Step 5: Train and evaluate models
print("\nModel Accuracy Comparison:")
for name, clsfr in models.items():
    clsfr.fit(X_train, y_train)
    y_pred = clsfr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.2f}")


def run_models(data):
    data.columns = data.columns.str.strip()
    data = data.fillna(data.mean(numeric_only=True))

    X = data.drop('TYPE', axis=1).select_dtypes(include='number')
    y = data['TYPE']
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True)
    }

    stacking_estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True)),
        ('nv', GaussianNB()),
        ('xb', XGBClassifier(eval_metric='logloss'))
    ]
    stack_model = StackingClassifier(estimators=stacking_estimators, final_estimator=RandomForestClassifier())
    models["Stacking Classifier"] = stack_model

    results = {}
    for name, clsfr in models.items():
        clsfr.fit(X_train, y_train)
        y_pred = clsfr.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = f"{acc:.2f}"

    return results

