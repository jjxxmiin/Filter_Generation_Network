import torch
import torch.nn as nn
import torch.nn.functional as F


class GFLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GFLayer, self).__init__()
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.weights = torch.nn.Parameter(torch.randn((out_ch, in_ch, 3)), requires_grad=True)

    def forward(self, x):
        w = torch.empty(self.out_ch, self.in_ch, 3, 3).to('cuda')

        for i in range(self.out_ch):
            for j in range(self.in_ch):
                w[i][j] = x[0].mul(self.weights[i][j][0]) + x[1].mul(self.weights[i][j][1]) + x[2].mul(self.weights[i][j][2])

        return w


class FGN(nn.Module):
    def __init__(self, filters):
        super(FGN, self).__init__()
        self.filters = filters

        kernel_size = filters.size(1)

        self.gf1 = GFLayer(3, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, stride=1, padding=1, bias=False)

        self.gf2 = GFLayer(32, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=1, padding=1, bias=False)

        self.classifier = nn.Sequential(nn.Linear(1568, 1024),
                                        nn.Linear(1024, 10))

    def forward(self, x):
        conv1_filters = self.gf1(self.filters)
        x = F.conv2d(x, conv1_filters, stride=1, padding=1)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        conv2_filters = self.gf2(self.filters)
        x = F.conv2d(x, conv2_filters, stride=1, padding=1)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    torch.manual_seed(20200504)
    torch.cuda.manual_seed(20200504)
    print(f"CPU seed : {torch.initial_seed()} GPU seed : {torch.cuda.initial_seed()}")

    img = torch.autograd.Variable(torch.ones(1, 3, 32, 32), requires_grad=True)
    filters = torch.autograd.Variable(torch.ones(3, 3, 3), requires_grad=True)

    model = FGN(filters)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(2):
        output = model(img)

        optimizer.zero_grad()
        output.sum().backward()
        optimizer.step()

    # for i in model.parameters():
    #     print(i.shape)