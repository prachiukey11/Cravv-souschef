import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        outs = [None] * len(laterals)
        outs[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(outs[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            outs[i] = laterals[i] + up
        return [s(o) for s, o in zip(self.smooth, outs)]


class SegHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, feats, out_size):
        p2, p3, p4, p5 = feats
        p3 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p5 = F.interpolate(p5, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([p2, p3, p4, p5], dim=1)
        x = self.conv(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


class ClsHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, feats):
        x = self.pool(feats[-1]).flatten(1)
        return self.fc(x)


class DetHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes * 5),
        )

    def forward(self, feats):
        _, p3, p4, p5 = feats
        x = torch.cat([self.pool(p).flatten(1) for p in (p3, p4, p5)], dim=1)
        x = self.fc(x).view(-1, self.num_classes, 5)
        conf = x[..., 0]
        box = torch.sigmoid(x[..., 1:])
        return conf, box


class MultiTaskModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.fpn = FPN([64, 128, 256, 512], 128)
        self.seg_head = SegHead(128, 2)
        self.det_head = DetHead(128, 3)
        self.cls_head = ClsHead(128, 5)

    def forward(self, x, task=None):
        c0 = self.stem(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        feats = self.fpn([c1, c2, c3, c4])

        out = {}
        if task is None or task == "seg":
            out["seg"] = self.seg_head(feats, x.shape[-2:])
        if task is None or task == "det":
            out["det"] = self.det_head(feats)
        if task is None or task == "cls":
            out["cls"] = self.cls_head(feats)
        return out
