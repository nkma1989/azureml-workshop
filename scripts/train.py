import argparse
import os
import joblib

from azureml.core import Dataset, Run
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier,plot_importance

from sklearn.metrics import accuracy_score,plot_confusion_matrix,plot_precision_recall_curve,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split

def main():
    """
    Trains a classification model for the Diabetes dataset
    """
    #Getting run context
    run = Run.get_context()
    ws = run.experiment.workspace
    #Parsing input
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--target_column", type=str)
    args = parser.parse_args()

    #Loading dataset
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    df = dataset.to_pandas_dataframe()
    #Forcing numerical schema
    df=df.apply(pd.to_numeric, errors='coerce')
    print(df.head())
    print(df.dtypes)
    #Defining labels
    target_column=args.target_column
    X = df.loc[:,df.columns != target_column]
    y=df[target_column]

    #Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    #Model params
    lr=0.09
    n_estimators=50
    max_depth=3
    #Logging model params
    run.log("Learning Rate", lr)
    run.log("n_estimators",n_estimators)
    run.log("Max Depth",max_depth)
    #XGB object
    model = XGBClassifier(
        learning_rate =lr,
        n_estimators=n_estimators,
        max_depth=max_depth,
        )
    #Fitting model
    model.fit(X_train, y_train)

    #Logging metrics
    run.log("Accuracy", accuracy_score(y_test, model.predict(X_test)))
    run.log("Precision Score", precision_score(y_test, model.predict(X_test)))
    run.log("Recall Score", recall_score(y_test, model.predict(X_test)))
    run.log("F1 score",f1_score(y_test, model.predict(X_test)))

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
    
    #Save Model 
    os.mkdir("./outputs/model")
    joblib.dump(model, "./outputs/model/model.joblib")
    #pickle.dump(model, open("./outputs/model/model.pkl", "wb"))
    #Hack to enable model registering within script
    ###https://stackoverflow.com/questions/58933565/how-to-register-model-from-the-azure-ml-pipeline-script-step
    run.upload_file("outputs/model/model.joblib","outputs/model/model.joblib")
    
    #Registering model
    run.register_model(
        model_name=args.model_name,
        model_path="outputs/model/model.joblib",
        description="A classification model",
        tags={'Learning Rate': lr, 
                'N_estimators': n_estimators, 
                'Max Depth': max_depth
                }
        )
    run.complete()
if __name__ == "__main__":
    main()
