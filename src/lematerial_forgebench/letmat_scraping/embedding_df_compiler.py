import pandas as pd
import glob
import pickle 

if __name__ == "__main__":
    mlips = ["orb", "mace"]
    dir_name = "test_small_lematbulk"
    for mlip in mlips: 
        full_df = []
        df_paths = glob.glob(                    
                        "data/"
                        + dir_name
                        + "/"
                        + mlip
                        + "/*"
                        )
        for path in df_paths:
            with open(path, "rb") as f:
                temp_df = pickle.load(f)
            if len(full_df) == 0:
                full_df = temp_df
            else:
                full_df = pd.concat([full_df, temp_df])
        
        full_df.to_pickle(
                        "data/"
                        + dir_name
                        + "/"
                        + mlip
                        + "_full_embedding_df.pkl"
                        )
            
