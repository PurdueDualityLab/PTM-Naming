
import dotenv, os, pickle

class ClusterDataset():

    def __init__(self, cluster_dir=None):
        if cluster_dir == None:
            dotenv.load_dotenv(".env", override=True)
            cluster_dir = os.getenv("CLUSTER_DIR")
        self.cluster_dir = cluster_dir
        self.vec_l = self.load_pkl(self.cluster_dir + "/vec_l.pkl")
        self.vec_p = self.load_pkl(self.cluster_dir + "/vec_p.pkl")
        self.vec_d = self.load_pkl(self.cluster_dir + "/vec_d.pkl")
        self.key_l = self.load_pkl(self.cluster_dir + "/k_l.pkl")
        self.key_p = self.load_pkl(self.cluster_dir + "/k_p.pkl")
        self.key_d = self.load_pkl(self.cluster_dir + "/k_d.pkl")

    def load_pkl(self, file_loc):
        with open(file_loc, 'rb') as f:
            return pickle.load(f)
    
    def get(self, vec_category, item_name, mode="arch"):

        if vec_category == "l":
            vec, key = self.vec_l, self.key_l
        elif vec_category == "p":
            vec, key = self.vec_p, self.key_p
        elif vec_category == "d":
            vec, key = self.vec_d, self.key_d
        else:
            raise ValueError("Invalid name of vector. Choose from l, p, or d.")

        if mode == "arch":
            model_vec_list = vec[item_name]
            recnstr_dict = {item_name: dict()}
            for model_name, model_vec in model_vec_list.items():
                recnstr_dict[item_name][model_name] = dict()
                for i in range(len(key)):
                    dict_key, dict_val = key[i], model_vec[i]
                    if dict_val == 0: continue
                    recnstr_dict[item_name][model_name][dict_key] = dict_val
            return recnstr_dict
        else:
            raise ValueError(f"Mode {mode} not supported.")
                
if __name__ == "__main__":

    ds = ClusterDataset()
    v = ds.get("l", "WavLMForCTC")
    print(v)