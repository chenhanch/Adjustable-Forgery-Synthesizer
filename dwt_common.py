import numpy as np


def dwt(x):
    """
    input x: h, w, c
    output : h/2, w/2, 4c
    """
    x01 = x[0::2, :, :] / 2
    x02 = x[1::2, :, :] / 2
    x1 = x01[:, 0::2, :]
    x2 = x02[:, 0::2, :]
    x3 = x01[:, 1::2, :]
    x4 = x02[:, 1::2, :]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return np.concatenate((x_LL, x_HL, x_LH, x_HH), axis=-1)


def iwt(x):
    """
    input x: h/2, w/2, 4c
    output : h, w, c
    """
    r = 2
    in_height, in_width, in_channel = x.shape
    #print([in_height, in_width, in_channel])
    out_channel, out_height, out_width = int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, :, 0:out_channel] / 2
    x2 = x[:, :, out_channel:out_channel * 2] / 2
    x3 = x[:, :, out_channel * 2:out_channel * 3] / 2
    x4 = x[:, :, out_channel * 3:out_channel * 4] / 2


    h = np.zeros([out_height, out_width, out_channel], dtype=np.float32)

    h[0::2, 0::2, :] = x1 - x2 - x3 + x4
    h[1::2, 0::2, :] = x1 - x2 + x3 - x4
    h[0::2, 1::2, :] = x1 + x2 - x3 - x4
    h[1::2, 1::2, :] = x1 + x2 + x3 + x4

    return h
    
        
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

