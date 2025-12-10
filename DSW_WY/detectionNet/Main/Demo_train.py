from Config import _C
from Train.trainer import train

if __name__ == '__main__':
    """
    使用时,请先打开visdom
    
    命令行 输入  pip install visdom          进行安装 
    输入        python -m visdom.server'    启动
    """
    if _C.TRAIN_STAGE==1:
        print('---------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------')
        print('STAGE1 TRAINING...')
        train()
    elif _C.TRAIN_STAGE==2:
        print('---------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------')
        print('STAGE2 TRAINING...')
        train()
    else:
        print('wrong setting training stage')