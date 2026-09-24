import torch

from model import MNISTCNN
from data import train_loader, test_loader


model = MNISTCNN()

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for images, labels in train_loader:

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = loss_fn(predictions, labels)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Average Loss: {average_loss:.4f}"
    )


# Evaluation

model.eval()

test_loss = 0.0
correct = 0

with torch.no_grad():

    for images, labels in test_loader:

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        test_loss += loss.item()

        predicted_labels = predictions.argmax(dim=1)

        correct += (predicted_labels == labels).sum().item()


average_test_loss = test_loss / len(test_loader)

accuracy = correct / len(test_loader.dataset)

print(f"Test Loss: {average_test_loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")