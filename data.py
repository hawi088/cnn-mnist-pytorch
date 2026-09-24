import torch
from torchvision import datasets
from torchvision import v2

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

print("Training dataset Length:",len(train_dataset))
print("Test dataset Length:", len(test_dataset))
image, label = train_dataset[0]
print(image)
print("Image type:",type(image))
print("Image size:", image.size)
print("Label:",label)


image.show()