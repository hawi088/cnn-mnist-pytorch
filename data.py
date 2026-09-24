from torchvision import datasets

train_dataset = datasets.MNIST(
    root="data",
    train = True,
    download=True
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True
)

print("Training dataset Length:",len(train_dataset))
print("Test dataset Length:", len(test_dataset))
image, label = train_dataset[0]
print(image)
print("Image type:",type(image))
print("Image size:", image.size)
print("Label:",label)