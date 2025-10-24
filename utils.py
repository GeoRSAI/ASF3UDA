import torch

def iou(output, gt):
    smooth = 1e-5
    output = output.view(-1)  # 需要赋值回变量
    gt = gt.view(-1)          # 需要赋值回变量
    pred_inds = output >= 0.5
    gt_inds = gt >= 0.5       # 使用相同的阈值
    intersection = (pred_inds & gt_inds).float().sum()  # 使用逻辑与
    union = (pred_inds | gt_inds).float().sum()         # 使用逻辑或
    iou = (intersection + smooth) / (union + smooth)
    return iou.item() * 100


def tp(output, target):
    pred_inds = output >= 0.5
    target_inds = target == 1
    res = float(pred_inds[target_inds].sum())
    return res


def fp(output, target):
    pred_inds = output >= 0.5
    target_inds = target != 1
    res = float(pred_inds[target_inds].sum())
    return res


def fn(output, target):
    pred_inds = output < 0.5
    target_inds = target == 1
    res = float(pred_inds[target_inds].sum())
    return res


def tf(output, target,):
    pred_inds = output < 0.5
    target_inds = target != 1
    res = float(pred_inds[target_inds].sum())
    return res


def f1(output, target):
    smooth = 1e-5
    # target = torch.argmax(target, dim=1)
    output.view(-1)
    target.view(-1)
    f1 = (2*tp(output, target) + smooth)/\
        (2*tp(output, target)+fp(output, target)+fn(output, target) + smooth)
    return f1*100

if __name__ == "__main__":
    x = torch.rand(2, 3, 128, 128).cuda()
    y = torch.rand(2, 3, 128, 128).cuda()
    a = iou(x, y)
    print(a)

