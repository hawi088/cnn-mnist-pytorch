import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True)
])
train_dataset = datasets.MNIST(
    root="data",
    train = True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# print("Training dataset Length:",len(train_dataset))
# print("Test dataset Length:", len(test_dataset))
# image, label = train_dataset[0]
# print(image)
# print("Image type:",type(image))
# print("Image size:", image.size)
# print("Label:",label)


# image.show()


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)