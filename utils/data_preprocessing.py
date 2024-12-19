import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from the given file path."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocess the data: handle missing values and standardize."""
    # Replace zeros in specific columns with NaN and fill with the median
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[columns_with_zeros] = data[columns_with_zeros].replace(0, pd.NA)
    data.fillna(data.median(), inplace=True)

    # Features and labels
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
