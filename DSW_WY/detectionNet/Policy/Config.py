from yacs.config import CfgNode as CN

_C=CN()

_C.DATASET_DIR='/home/wy/dataset/topicData/MainProject'     #数据集根目录
_C.CURRENT_PROJECT_DIR='/home/wy/py_doc/MainProject/Part2/Policy'       #当前工程根目录
_C.PRETRAINED_MODEL_PATH='/home/wy/py_doc/MainProject/Part1/Weight/trained/mobile/mobilenet_best.pth'

_C.IMAGE_SIZE=512
#_C.INPUT_SIZE=224
_C.NUM_CLASSES=3
_C.CLASSES=['Cystic','Odontoma','OssifyingFibroma']

_C.BATCH_SIZE=9

_C.LEARNING_RATE=0.003
_C.DATASET_NUM_WORKERS = 4
_C.DATASET_PIN_MEMORY = True

_C.DEVICE = 'cuda:0' 
_C.EPOCH=10

_C.STATE_DIM=9
_C.HIDDEN_DIM=128
_C.ACTION_CARD=9

_C.CRITIC_LR = 5e-3
_C.ACTOR_LR = 2e-3
_C.GAMMA=0.9
_C.AC_EPOCHS=10