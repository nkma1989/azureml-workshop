import argparse
import os
from azureml.core import Dataset, Run
import pandas as pd

def main():
    """
    Data pre processing of the diabetes dataset
    """
    
    #Getting run context
    run = Run.get_context()
    ws = run.experiment.workspace
    #Parsing input
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    #Loading dataset
    dataset = Dataset.get_by_id(ws, id=args.input_data)
    df = dataset.to_pandas_dataframe()
    print(df.columns)

    #Renaming columns
    df.rename(columns={'Y':'Target'},inplace=True)
    print(df.head())

    #Creating binary target
    df['Binary_Target']=pd.cut(df['Target'], 2,labels=[0,1])
    #Dropping target
    df.drop(columns=['Target'],inplace=True)

    #Save to local path
    #df.to_csv('./data/train.csv')
    
    mounted_output_path = args.output
    os.makedirs(mounted_output_path, exist_ok=True)
    df.to_csv(mounted_output_path+'/train.csv',index=False)
    
    run.complete()
if __name__ == "__main__":
    main()
