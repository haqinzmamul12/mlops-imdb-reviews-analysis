import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import yaml
from src.data.load_data import load_imdb_data 

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class DataTransformation:
    def __init__(self):
        self.STOPWORDS = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        with open("config/config.yaml", 'r') as f:
                config =yaml.safe_load(f)
        self.output_path =config['data']['interim_path']

    def clean_text(self, text: str):
        try:
            text =text.lower()
            text = re.sub(r"<.*?>", " ", text) 
            text = re.sub(r"[^a-z\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            words = text.split()
            words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.STOPWORDS
                ]
            
            text = " ".join(words)
            return text 
        
    
        except Exception as e:
            print(f"Error Occured: {repr(e)}")

    def preprocess_data(self):
        try:
            print("collecting raw dataset...")
            df_raw =load_imdb_data("raw")
            data =df_raw.copy() 
            data.drop_duplicates() 
            data.dropna() 
            print("Processing raw data...")
            data['review'] =data['review'].apply(self.clean_text) 
            data.to_csv(self.output_path, index=False)
            print(f"cleaned dataset saved to {self.output_path}")

        except Exception as e:
            print(f"Error Occured: {repr(e)}")


