from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/random_forest_model.pkl')  # Save model
    return rf

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
