import pandas as pd 
import yaml

def load_imdb_data(file_name: str):
    try:
        with open("config/config.yaml", 'r') as f:
            config =yaml.safe_load(f)

        input_path =""
        if file_name=="raw":
            input_path =config['data']['raw_path']
        elif file_name=='interim':
            input_path =config['data']['interim_path']
        else:
            input_path =config['data']['cleaned_path'] 

        df =pd.read_csv(input_path)
        print(f"{file_name} dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"Error encountered: {repr(e)}")
