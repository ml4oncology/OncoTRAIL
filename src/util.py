import pickle

def load_pickle(save_dir: str, filename: str, err_msg=None):
    filepath = f'{save_dir}/{filename}.pkl'
    with open(filepath, 'rb') as file:
        output = pickle.load(file)
    return output

def save_pickle(result, save_dir: str, filename: str):
    filepath = f'{save_dir}/{filename}.pkl'
    with open(filepath, 'wb') as file:    
        pickle.dump(result, file)