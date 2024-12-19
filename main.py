from utils.data_preprocessing import load_data, preprocess_data
from models.traditional_ml import train_random_forest, evaluate_model
from models.neural_networks import build_nn, train_nn
from utils.evaluation import plot_roc_curve
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/diabetes.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train traditional ML model
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    # Train Neural Network
    print("Training Neural Network...")
    nn_model = build_nn(input_dim=X_train.shape[1])
    nn_model = train_nn(nn_model, X_train, y_train, X_test, y_test)

    # Evaluate Neural Network
    y_probs = nn_model.predict(X_test).ravel()
    print(f"Neural Network AUC: {roc_auc_score(y_test, y_probs):.2f}")
    plot_roc_curve(y_test, y_probs)
