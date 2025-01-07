import os

# Chemins
base_dir = os.path.dirname(__file__)

#data
data_path = os.path.join(base_dir, "data")


# preprocessed data number 2
preprocessed_data2 = os.path.join(base_dir, "preprocessed_data2") 

# SBERT model path
sbert_model_path = os.path.join(base_dir, "models\paraphrase-MiniLM-L6-v2")