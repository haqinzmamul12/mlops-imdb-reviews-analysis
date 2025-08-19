import pickle


def dump_model(path, model):
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
            print(f"model dumped to {path}")
    except Exception as e:
        print(f"Error Encountered: {repr(e)}")
