from utils.test_model import test_model
from utils.train_model import train_model

folders_to_use = ["CSSD", "ECSSD", "MSRA-B", "MSRA10K_Imgs_GT", "THuR15000"]

x_path = "./PseudoSaliency_avg_release/Images/MSRA-B/Imgs"
y_path = "./PseudoSaliency_avg_release/Maps/MSRA-B/Imgs"

train_model("model8_less_upsampling.h5", x_path, y_path)

# test_model("saved_models/checkpoints/model.10.h5",
#            "./PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/10354.jpg",
#            "./PseudoSaliency_avg_release/Maps/MSRA10K_Imgs_GT/Imgs/10354.jpg")

print("#### DONE ####")
