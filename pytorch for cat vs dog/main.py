import torch
from torch.autograd import Variable
from data.dataset import DogCat
from torch.utils.data import DataLoader
import models
from config import DefaultConfig

opt = DefaultConfig()


def train():
    # 1 数据
    train_data = DogCat(opt.train_data_root)
    trainloader = DataLoader(train_data, batch_size=opt.batch_size)

    val_data = DogCat(opt.val_data_root, train=False)


    # 2 模型

    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    for epoch in range(opt.max_epoch):
        for ii, (data, label) in enumerate(trainloader):
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            if ii % 20 == 0:
                print('epochs[{}] loss {}'.format(str(epoch),str(loss.data[0])))

    model.save()


if __name__ == '__main__':
    train()
