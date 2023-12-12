import os

path = "../PseudoSaliency_avg_release/Images/MSRA-B/"
move_to_path = "../PseudoSaliency_avg_release/Images/MSRA-B/pngs/"

for file in os.listdir(path):
    if file.endswith('.png'):
        os.rename(path + file, move_to_path + file)