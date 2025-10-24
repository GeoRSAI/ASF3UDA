import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

urban_dic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/LoveAD/Urban"
rural_dic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/LoveAD/Rural"
whu_dic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/WHU/AerialImage"
inr_dic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/inria"
pos_dic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/Potsdam"
vah_dic = "/mnt/334a6b76-a5f7-4669-8ac2-daacd80b93ba/dataset/Vaihingen"

WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
POWER = 0.9
LEARNING_RATE_D = 1e-4
LAMBDA_ADV_MAIN = 0.001
LAMBDA_ADV_AUX = 0.01
LAMBDA_DECOMP = 0.5
LAMBDA_SEG_MAIN = 1.0
LAMBDA_SEG_AUX = 1.0
LAMBDA_SEG_COVAR = 0.01
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
NUM_WORKER = 1
NUM_EPOCH = 100
OUT_ADV = True
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_LOAD_NAME = ""
MODEL_SAVE_NAME = ""



