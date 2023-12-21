from utils.test_model import test_model

model_path = "../saved_models/checkpoints/model.10.h5"

test_model(model_path,
           "../../SP1_data/PseudoSaliency_avg_release/Images/MSRA-B/Imgs/1_25_25246.jpg",
           "../../SP1_data/PseudoSaliency_avg_release/Maps/MSRA-B/Imgs/1_25_25246.jpg")

test_model(model_path,
           "../../SP1_datasalicon/stimuli/train/COCO_train2014_000000008014.jpg",
           "../../SP1_datasalicon/saliency/train/COCO_train2014_000000008014.png")

test_model(model_path,
           "../../SP1_datasalicon/stimuli/train/COCO_train2014_000000005011.jpg",
           "../../SP1_datasalicon/saliency/train/COCO_train2014_000000005011.png")

test_model(model_path,
           "../../SP1_data/PseudoSaliency_avg_release/Images/ECSSD/images/0552.jpg",
           "../../SP1_data/PseudoSaliency_avg_release/Images/ECSSD/images/0552.jpg")
