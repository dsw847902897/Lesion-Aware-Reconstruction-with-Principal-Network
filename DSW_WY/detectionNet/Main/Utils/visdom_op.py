import visdom
import torch
import numpy as np


def setup_visdom(**kwargs):

    """
    eg :
        vis_eval = setup_visdom(env='SSD_eval')

    :param kwargs:
    :return:
    """
    vis = visdom.Visdom(**kwargs)
    return vis


def visdom_line(vis, y, x, win_name, update='append'):

    """
    eg :
        visdom_line(vis_train, y=[loss], x=iteration, win_name='loss')
    """
    if not isinstance(y,torch.Tensor):
        y=torch.Tensor(y)
    y = y.unsqueeze(0)
    x = torch.Tensor(y.size()).fill_(x)
    vis.line(Y=y, X=x, win=win_name, update=update, opts={'title':win_name})
    return True


def visdom_images(vis, images,win_name,num_show=None,nrow=None):
    """
    eg:
        visdom_images(vis_train, images, num_show=3, nrow=3, win_name='Image')

    """
    if not num_show:
        num_show = 6
    if not nrow:
        nrow = 3
    num = images.size(0)
    if num > num_show:
        images = images [:num_show]
    vis.images(tensor=images,nrow=nrow,win=win_name)
    return True


def visdom_image(vis, image,win_name):
    """
    eg :
        visdom_image(vis=vis, image=drawn_image, win_name='image')

    """
    vis.image(img=image, win=win_name)
    return True

def visdom_bar(vis, X, Y, win_name):
    """
    绘制柱形图
    eg:
        visdom_bar(vis_train, X=cfg.DATASETS.CLASS_NAME, Y=ap, win_name='ap', title='ap')

    """
    dic = dict(zip(X,Y))
    del_list = []
    for val in dic:
        if np.isnan(dic[val]):
            del_list.append(val)

    for val in del_list:
        del dic[val]

    vis.bar(X=list(dic.values()),Y=list(dic.keys()),win=win_name, opts={'title':win_name})
    return True