from yacs.config import CfgNode as CN
import os


_C = CN()

############## sharing part ##############
_C.CURRENT_DIR='/home/wy/py_doc/MainPart2/detectionNet'
_C.DEVICE = 'cuda:1'  

_C.EMERGE_RATIO=0.05
_C.TRAIN_STAGE=2   #1 or 2 or 3
_C.STATE_POINTS=3

################ retina ##################
_C.RESNET_PRETRAIN_DIR=os.path.join(_C.CURRENT_DIR,'Weight','pretrained')
_C.DATASET_DIR1 = '/home/wy/dataset/topicData/MainProject'
_C.RETINA_SAVE_DIR=os.path.join(_C.CURRENT_DIR,'Weight','trained','retina')

_C.IMAGE_SIZE1 = 580
_C.CLASS_NAMES1 = ('__background__','Odontoma','OssifyingFibroma', 'Cystic')
_C.NUM_CLASSES1=4

_C.BASEMODEL = 'resnet50' #resnet18, resnet34, resnet50, resnet101, resnet152

_C.DATA_PIXEL_MEAN = [0, 0, 0]
_C.DATA_PIXEL_STD = [1, 1, 1]

_C.ANCHORS_NUMS=9
_C.ANCHORS_FEATURE_MAPS = [(73, 73), (37, 37), (19, 19), (10, 10), (5, 5)]
_C.ANCHORS_SIZES =[32, 64, 128, 256, 512]
_C.ANCHORS_RATIOS = [0.5, 1, 2]
_C.ANCHORS_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
_C.ANCHORS_CLIP = True
_C.ANCHORS_CENTER_VARIANCE = 0.1
_C.ANCHORS_SIZE_VARIANCE = 0.2 
_C.ANCHORS_THRESHOLD = 0.5

_C.TEST_CONFIDENCE_THRESHOLD=0.5
_C.TEST_MAX_PER_CLASS = -1    
_C.TEST_MAX_PER_IMAGE = 1     
_C.NEG_POS_RATIO = 4 

_C.RETINA_VIS_STEP = 10           # visdom可视化训练过程,打印步长
_C.RETINA_SAVE_STEP = 100 
_C.MAX_ITER=15000
_C.LEARNING_RATE1 = 1e-3 
_C.MOMENTUM1 = 0.9         # 优化器动量.默认优化器为SGD
_C.WEIGHT_DECAY1 = 5e-4
_C.BATCH_SIZE1=12
_C.SHUFFLE1=True
_C.NUM_WORKERS1=4
_C.SCHEDULER_GAMMA = 0.1  # 学习率衰减率
_C.SCHEDULER_LR_STEPS = [80000, 100000]

_C.EVAL_BATCH_SIZE1=48
_C.EVAL_SHUFFLE1=False
_C.EVAL_NUM_WORKERS1=4


_C.MULTIBOXLOSS_ALPHA = 0.25
_C.MULTIBOXLOSS_GAMMA = 2  

################ mobile ##################
_C.MOBILE_TRAINED_DIR=os.path.join(_C.CURRENT_DIR,'Weight','trained','mobile')
_C.MOBILE_PICK='mobilenet_best.pth'

_C.IMAGE_SIZE2=512
_C.CLASS_NAMES2 = ('Cystic','Solid','Mixed')
_C.NUM_CLASSES2=3

################# final eval ################
_C.EVAL_BEGIN_MODEL=91
_C.EVAL_END_MODEL=92

_C.DETECT_RETINA='model_5000.pkl'
_C.DETECT_SAVE_PATH=os.path.join(_C.CURRENT_DIR,'DetectResults')

####################### AC网络参数 ########################
_C.STATE_DIM=9
_C.HIDDEN_DIM=128
_C.ACTION_CARD=9

_C.CRITIC_LR = 5e-3
_C.ACTOR_LR = 2e-3
_C.GAMMA=0.9

_C.AC_PICK=True