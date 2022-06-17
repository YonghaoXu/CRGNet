import numpy as np

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
 

def index2bgr_z(c_map):

    # mapping W x H x 1 class index to W x H x 3 BGR image
    im_col, im_row = np.shape(c_map)
    c_map_r = np.ones((im_col, im_row), 'uint8')*255
    c_map_g = np.ones((im_col, im_row), 'uint8')*255
    c_map_b = np.ones((im_col, im_row), 'uint8')*255
    c_map_r[c_map == 0] = 0
    c_map_r[c_map == 1] = 100
    c_map_r[c_map == 2] = 0
    c_map_r[c_map == 3] = 0
    c_map_r[c_map == 4] = 150
    c_map_r[c_map == 5] = 0
    c_map_r[c_map == 6] = 255
    c_map_r[c_map == 7] = 150
    c_map_g[c_map == 0] = 0
    c_map_g[c_map == 1] = 100
    c_map_g[c_map == 2] = 125
    c_map_g[c_map == 3] = 255
    c_map_g[c_map == 4] = 80
    c_map_g[c_map == 5] = 0
    c_map_g[c_map == 6] = 255
    c_map_g[c_map == 7] = 150
    c_map_b[c_map == 0] = 0
    c_map_b[c_map == 1] = 100
    c_map_b[c_map == 2] = 0
    c_map_b[c_map == 3] = 0
    c_map_b[c_map == 4] = 0
    c_map_b[c_map == 5] = 150
    c_map_b[c_map == 6] = 0
    c_map_b[c_map == 7] = 255
    c_map_rgb = np.zeros((im_col, im_row, 3), 'uint8')
    c_map_rgb[:, :, 0] = c_map_r
    c_map_rgb[:, :, 1] = c_map_g
    c_map_rgb[:, :, 2] = c_map_b
    
    return c_map_rgb

def index2bgr_v(c_map):

    # mapping W x H x 1 class index to W x H x 3 BGR image
    im_col, im_row = np.shape(c_map)
    c_map_r = np.ones((im_col, im_row), 'uint8')*255
    c_map_g = np.ones((im_col, im_row), 'uint8')*255
    c_map_b = np.ones((im_col, im_row), 'uint8')*255
    c_map_r[c_map == 0] = 255
    c_map_r[c_map == 1] = 0
    c_map_r[c_map == 2] = 0
    c_map_r[c_map == 3] = 0
    c_map_r[c_map == 4] = 255
    c_map_g[c_map == 0] = 255
    c_map_g[c_map == 1] = 0
    c_map_g[c_map == 2] = 255
    c_map_g[c_map == 3] = 255
    c_map_g[c_map == 4] = 255
    c_map_b[c_map == 0] = 255
    c_map_b[c_map == 1] = 255
    c_map_b[c_map == 2] = 255
    c_map_b[c_map == 3] = 0
    c_map_b[c_map == 4] = 0
    
    c_map_rgb = np.zeros((im_col, im_row, 3), 'uint8')
    c_map_rgb[:, :, 0] = c_map_r
    c_map_rgb[:, :, 1] = c_map_g
    c_map_rgb[:, :, 2] = c_map_b
    
    return c_map_rgb


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class=19):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def eval_image(predict,label,num_classes):
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return TP,FP,TN,FN,len(label)