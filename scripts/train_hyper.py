import argparse
import joblib
import os
from azureml.core import Dataset, Run
import matplotlib.pyplot as plt
from xgboost import XGBClassifier,plot_importance
from sklearn.metrics import accuracy_score,plot_confusion_matrix,plot_precision_recall_curve,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

def main():
    """
    Hyper parameter tuning for the diabetes dataset
    """
    #Getting run context
    run = Run.get_context()
    ws = run.experiment.workspace
    #Parsing input
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    parser.add_argument("--target_column", type=str)
    #Parameter to tune
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--max_depth", type=int)
    #parser.add_argument('--output_dir', type=str, help='output directory')

    args = parser.parse_args()
    #Loading dataset
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    df = dataset.to_pandas_dataframe()

    #Defining labels
    target_column=args.target_column
    X = df.loc[:,df.columns != target_column]
    y=df[target_column]

    #Using preset split to determine train and test data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    #Logging model params
    run.log("Learning Rate", args.learning_rate)
    run.log("n_estimators",args.n_estimators)
    run.log("Max Depth",args.max_depth)

    #Fitting model
    model = XGBClassifier(
            learning_rate =args.learning_rate,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            )
    model.fit(X_train, y_train)
    #Logging metrics
    run.log("Accuracy", accuracy_score(y_test, model.predict(X_test)))
    run.log("Precision_Score", precision_score(y_test, model.predict(X_test)))
    run.log("Recall_Score", recall_score(y_test, model.predict(X_test)))
    run.log("F1_score",f1_score(y_test, model.predict(X_test)))

    #Saving figures
    plot_confusion_matrix(model, X_test, y_test)
    plt.savefig("outputs/confusion_matrix.png")
    run.log_image("confusion_matrix", "outputs/confusion_matrix.png")
    #PR curve
    plot_precision_recall_curve(model,X_test,y_test)
    plt.savefig("outputs/pr_curve.png")
    run.log_image("Precision Recall Curve", "outputs/pr_curve.png")
    #Feature importance
    plot_importance(model)
    plt.savefig("outputs/feature_imp.png")
    run.log_image("Feature Importance", "outputs/feature_imp.png")
    #Saving Model
    os.mkdir("outputs/model")
    joblib.dump(model, "outputs/model/model.joblib")

    run.complete()

if __name__ == "__main__":
    main()
