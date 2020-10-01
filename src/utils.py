import numpy as np
import pandas as pd


def calc_chlac_dev(div_data, masks, mask_n):
    chlacs = np.array([])
    for m, mn in zip(masks, mask_n):
        chlacs = np.append(
            chlacs, (np.sum(np.sum(np.logical_and(m, div_data), axis=1) == mn))
        )
    return chlacs


# old version, too slow
def calc_chlac(div_data, masks, mask_n):
    chlacs = []
    for m, mn in zip(masks, mask_n):
        logic = np.logical_and(m, div_data)
        logic_sum = np.sum(logic, axis=1)
        chlacs.append(np.sum(logic_sum == int(mn)))
    return np.array(chlacs)


def split2boxel(data):
    x, y = data[0].shape
    d = 3
    li = []
    # 最外周は外してnp.whereを実行
    # np.whereは切り出したndarrayに対して条件に一致するインデックスを返す
    idxs_x, idxs_y = np.where(data[1, 1:-1, 1:-1] == 255)
    # i, jは切り出すボクセルの左上隅のインデックスを表す
    for i, j in zip(idxs_x, idxs_y):
        li.append(data[:, i: i + d, j: j + d])
    div_data = np.array(li).reshape([len(li), 27]).astype(np.int32)

    return div_data


# old version, too slow
def split2boxel_(data):
    x, y = data[0].shape
    d = 3
    li = []
    for i in range(x - 2):
        for j in range(y - 2):
            if data[1, i + 1, j + 1] != 0:
                li.append(data[:, i: i + d, j: j + d])
    div_data_ = np.array(li).reshape([len(li), 27]).astype(np.int32)

    return div_data_


def prepare_masks_chlac(mask_filepath):
    data = pd.read_csv(mask_filepath, header=None, index_col=0)
    N_mask = data.shape[0]

    masks = []
    for j in range(N_mask):
        mask = np.zeros([3, 3, 3])  # voxel
        _ = []
        for i in range(3):
            if data.iloc[j, i] == "x":
                _.append([])
            elif data.iloc[j, i] != "x":
                if len(data.iloc[j, i]) == 1:
                    _.append([data.iloc[j, i]])
                else:
                    _.append(data.iloc[j, i].split(","))

        for i in range(3):
            if len(_[i]) == 0:
                continue
            for m in _[i]:
                if m == "a":
                    mask[0, i, 0] = 1
                elif m == "b":
                    mask[1, i, 0] = 1
                elif m == "c":
                    mask[2, i, 0] = 1
                elif m == "d":
                    mask[0, i, 1] = 1
                elif m == "e":
                    mask[1, i, 1] = 1
                elif m == "f":
                    mask[2, i, 1] = 1
                elif m == "g":
                    mask[0, i, 2] = 1
                elif m == "h":
                    mask[1, i, 2] = 1
                elif m == "i":
                    mask[2, i, 2] = 1
        mask = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2])
        masks.append(mask)
    masks = np.array(masks)
    mask_n = np.array(masks).sum(axis=1)

    return masks, mask_n


def read_testdata():
    imgs = 255 * np.array(
        [
            [
                [1, 0, 1, 1, 0, 1],
                [1, 1, 0, 0, 1, 1],
                [0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0],
            ],
            [
                [1, 1, 1, 0, 0, 1],
                [1, 1, 1, 0, 0, 1],
                [0, 1, 1, 0, 1, 0],
                [1, 1, 0, 1, 1, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            [
                [1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1],
                [0, 1, 0, 1, 0, 1],
                [1, 1, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 1, 1, 0, 1, 1],
            ],
        ]
    )

    return imgs


def calc_hlac_dev(img, dim=3):
    masks_origin = [
        "000010000",
        "000011000",
        "001010000",
        "010010000",
        "100010000",
        "000111000",
        "001010100",
        "010010010",
        "100010001",
        "001110000",
        "010010100",
        "100010010",
        "000110001",
        "000011100",
        "001010010",
        "010010001",
        "100011000",
        "010110000",
        "100010100",
        "000110010",
        "000010101",
        "000011010",
        "001010001",
        "010011000",
        "101010000",
    ]
    masks = []
    masks_n = []

    # make masks
    for mask_bin in masks_origin:
        m = []
        s = 0
        for ch in mask_bin:
            m.append(int(ch))
            if int(ch) == 1:
                s += 1
        masks.append(np.array(m).reshape((3, 3)))
        masks_n.append(s)

    height, width = img.shape

    a = patchify(img, (3, 3))
    a = a.reshape(
        (height - 2) * (width - 2), 3, 3
    )  # (x-2, y-2, 3) -> ((x-2)*(y-2), 3, 3)

    ret = []
    for mask, n in zip(masks, masks_n):
        res = np.logical_and(mask, a)
        logic = np.sum(np.sum(res, axis=2), axis=1)
        ret.append(np.sum(logic == n))

    return ret


def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X - x + 1), (Y - y + 1), x, y)  # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize * np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
