import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from dataset import CarDateSet
from resnet import resnet50, resnet34
import argparse
from CenterLoss import CenterLoss
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self,  num_class, center_dim):
        super(Net, self).__init__()
        # self.backbone = resnet50(num_classes=2)
        self.ip1 = resnet50(num_classes=center_dim)
        self.ip2 = nn.Linear(center_dim, num_class, bias=False)

    def forward(self, x):
        ip1 = self.ip1(x)
        # ip1 = ip1.view(-1, 2)
        # ip1 = self.ip1(x)
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2, dim=1)

def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_datasets = CarDateSet('./data/train', './data/train.txt', transforms=None)

    test_datasets = CarDateSet('./data/train', './data/valid.txt', transforms=None)

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                              batch_size=args.batch_size,
                                              shuffle=True)

    print("Train numbers:{:d}".format(len(train_datasets)))

    # if args.pretrained:
    #     model = resnet50(num_classes=1000)
    #     model.load_state_dict(torch.load(args.pretrained_model))
    #     channel_in = model.fc.in_features  # 获取fc层的输入通道数
    #     # 然后把resnet的fc层替换成自己分类类别的fc层
    #     model.fc = nn.Linear(channel_in, args.num_class)
    # else:
    model = Net(num_class=args.num_class, center_dim=args.center_dim)
    print(model)
    # NLLLoss
    nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
    # CenterLoss
    loss_weight = args.loss_weight
    centerloss = CenterLoss(args.num_class, args.center_dim).to(device)
    # cost
    model = model.to(device)
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # start time
        start = time.time()
        index = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            ip1, pred = model(images)
            n_loss = nllloss(pred, labels)
            c_loss = centerloss(labels, ip1)
            loss = n_loss + loss_weight * c_loss

            if index % 10 == 0:
                if torch.cuda.is_available:
                    print('loss=%f, n_loss=%f,c_loss=%f' % (loss.data.cpu().numpy(),
                                                            n_loss.data.cpu().numpy(),
                                                            c_loss.data.cpu().numpy()))
                else:
                    print('loss=%f, n_loss=%f,c_loss=%f' % (loss.data.numpy(),
                                                            n_loss.data.numpy(),
                                                            c_loss.data.numpy()))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += 1


        if epoch % 1 == 0:
            end = time.time()
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss.item(), (end-start) * 2))

            model.eval()

            correct_prediction = 0.
            total = 0
            for images, labels in test_loader:
                # to GPU
                images = images.to(device)
                labels = labels.to(device)
                # print prediction
                ip1, pred = model(images)
                # equal prediction and acc

                _, predicted = torch.max(pred.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            print("Acc: %.4f" % (correct_prediction / total))

        # Save the model checkpoint
        torch.save(model, os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch)))
    print("Model save to %s."%(os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--center_dim", default=2, type=int)
    parser.add_argument("--loss_weight", default=1, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    # parser.add_argument("--net", default='resnet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='car', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    # parser.add_argument("--pretrained", default=False, type=bool)
    # parser.add_argument("--pretrained_model", default='./model/resnet50.pth', type=str)
    args = parser.parse_args()

    main(args)
