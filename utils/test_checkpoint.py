from utils.test_model import test_model

test_model("../saved_models/model4.h5",
           "../PseudoSaliency_avg_release/Images/MSRA10K_Imgs_GT/Imgs/10283.jpg",
           "../PseudoSaliency_avg_release/Maps/MSRA10K_Imgs_GT/Imgs/10283.jpg")