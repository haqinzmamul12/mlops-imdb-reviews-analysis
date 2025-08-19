import pickle


def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error Encountered: {repr(e)}")
