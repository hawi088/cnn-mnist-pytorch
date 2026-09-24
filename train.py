import torch
from model import MNISTCNN
from data import train_loader

model = MNISTCNN()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
epochs = 5
for epoch in range (epochs):
    model.train()
    total_loss = 0.0
    for images,labels in train_loader:
        #forward pass
        predictions = model(images)
        #calculate loss
        loss = loss_fn(predictions,labels)
        # Save one parameter before update
        before = model.network[0].weight[0, 0, 0, 0].item()

        #clear old gradient
        optimizer.zero_grad()
        #backpropagation
        loss.backward()
        # Inspect gradient
        gradient = model.network[0].weight.grad[0, 0, 0, 0].item()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Average Loss: {average_loss:.4f}"
    )