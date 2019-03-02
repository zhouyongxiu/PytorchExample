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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    model = resnet50(num_classes = args.num_class)
    print(model)
    # cost
    model = model.to(device)
    cost = nn.CrossEntropyLoss().to(device)
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
            outputs = model(images)
            loss = cost(outputs, labels)

            if index % 10 == 0:
                print (loss)
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
                outputs = model(images)
                # equal prediction and acc

                _, predicted = torch.max(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            print("Acc: %.4f" % (correct_prediction / total))

        # Save the model checkpoint
        torch.save(model, os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch)))
    print("Model save to %s."%(os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    # parser.add_argument("--net", default='resnet50', type=str)
    # parser.add_argument("--depth", default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--model_name", default='car', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    args = parser.parse_args()

    main(args)
