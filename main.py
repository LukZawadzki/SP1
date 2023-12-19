from utils.test_model import test_model
from utils.train_model import train_model
from utils.mlnet import create_ml_net

folders_to_use = ["CSSD", "ECSSD", "MSRA-B", "MSRA10K_Imgs_GT", "THuR15000"]

x_path = "../SP1_data/PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs"
y_path = "../SP1_data/PseudoSaliency_avg_release/Maps/MSRA10K_Imgs_GT/Imgs"

train_model("autoencoder.h5", x_path, y_path)

# test_model("saved_models/checkpoints/model.10.h5",
#            "../SP1_data/PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/0_7_7677.jpg",
#            "../SP1_data/PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/0_7_7677.jpg")

print("#### DONE ####")
