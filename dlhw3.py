import torch
from dataset import CUB
import transform
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter
import glob
import re
if __name__ == '__main__':
    print(torch.cuda.is_available())
    writer = SummaryWriter('runs/experiment_1/train_5')
    NUM_EPOCHS = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PRINT_INTERVAL = 100
    IMAGE_SIZE = 448
    TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
    TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
    TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
    TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

    path = "C:/Users/win11/Desktop/CUB_200_2011"

    train_transforms = transform.Compose([
        transform.ToCVImage(),
        transform.RandomResizedCrop(IMAGE_SIZE),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

    test_transforms = transform.Compose([
        transform.ToCVImage(),
        transform.Resize(IMAGE_SIZE),
        transform.ToTensor(),
        transform.Normalize(TEST_MEAN, TEST_STD)
    ])

    train_dataset = CUB(
        path,
        train=True,
        transform=train_transforms,
        target_transform=None
    )
    print(len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    test_dataset = CUB(
        path,
        train=False,
        transform=test_transforms,
        target_transform=None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    print(len(train_dataloader))
    print(len(test_dataloader))

    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    model = model.to(device)
    fc_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    optimizer = optim.SGD([
        {'params': base_params},
        {'params': model.fc.parameters(), 'lr': 0.01}
    ], lr=0.001, weight_decay=0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    accuracy_list = []

    checkpoint_files = glob.glob(
        "C:/Users/win11/PycharmProjects/pythonProject/checkpoint_2/hw2.1.1_checkpoint_epoch_*.pt")  # 获取所有的checkpoint文件
    checkpoint_file_epochs = [int(re.findall(r"\d+", file)[-1]) for file in checkpoint_files]  # 提取每个文件名中的epoch数值
    if checkpoint_file_epochs:  # 如果存在checkpoint文件
        valid_checkpoints = [epoch for epoch in checkpoint_file_epochs if epoch % 10 == 9]
        if valid_checkpoints:
            latest_checkpoint_epoch = max(valid_checkpoints)  # 找到最大的、符合epoch是9倍数规则的epoch数值，这是最近的有效checkpoint
            latest_checkpoint_file = f"C:/Users/win11/PycharmProjects/pythonProject/checkpoint_2/hw2.1.1_checkpoint_epoch_{latest_checkpoint_epoch}.pt"  # 生成最近的有效checkpoint的文件名
            checkpoint = torch.load(latest_checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = latest_checkpoint_epoch + 1  # 从下一个epoch开始训练
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % PRINT_INTERVAL == PRINT_INTERVAL - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / PRINT_INTERVAL))
                writer.add_scalar('training loss', running_loss / PRINT_INTERVAL, epoch * len(train_dataloader) + i)
                running_loss = 0.0
        scheduler.step()

        if epoch % 5 == 4:
            correct = 0
            total = 0
            running_loss_val = 0.0
            with torch.no_grad():
                model.eval()
                for data in test_dataloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss_val = criterion(outputs, labels)
                    running_loss_val += loss_val.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            writer.add_scalar('validation loss', running_loss_val/total, epoch)
            writer.add_scalar('validation accuracy', 100 * correct / total, epoch)
            model.train()
            accuracy = 100 * correct / total
            print('Accuracy of the network on the test set after epoch %d: %d %%' % (epoch, accuracy))
            accuracy_list.append(accuracy)
        if epoch % 10 == 9:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"C:/Users/win11/PycharmProjects/pythonProject/checkpoint_1/hw2.1.1_checkpoint_epoch_{epoch}.pt")
    torch.save(model.state_dict(), "C:/Users/win11/PycharmProjects/pythonProject/hw2.1.1.pt")
    print('Finished Training')
    writer.close()