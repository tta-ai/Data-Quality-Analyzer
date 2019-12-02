import pickle

def save_list(data, path):
    with open(path, "wb") as fp:   #Pickling
        pickle.dump(data, fp)
        
def load_list(path):
    with open(path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
        return b
    
    
