from utils.test_model import test_model

test_model("../saved_models/checkpoints/model.04.h5",
           "../../SP1_data/PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/5132.jpg",
           "../../SP1_data/PseudoSaliency_avg_release/Maps/MSRA10K_Imgs_GT/Imgs/5132.jpg")
