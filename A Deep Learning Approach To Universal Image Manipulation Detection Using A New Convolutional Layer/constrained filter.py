epochs = 30
steps = 0

train_losses, val_losses = [], []
train_accs, valid_accs = [], []

class_correct = torch.zeros((epochs, 4))
class_total = torch.zeros((epochs, 4))

for e in range(epochs):
    running_loss = 0
    train_accuracy = 0

    model.train()

    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        for i in range(model.convRes.weight.shape[0]):
            tmp = model.convRes.weight.detach()

            tmp[i, :, 2, 2] = torch.FloatTensor([[0]])
            tmp[i, :, :, :] /= sum(tmp[i, :, :, :].flatten())
            tmp[i, :, 2, 2] = torch.FloatTensor([[-1]])

        logits = model(data)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_loss = 0
    valid_accuracy = 0

    model.eval()
    with torch.no_grad():
        for data, labels in valid_loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            val_loss += criterion(logits, labels).item()

    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(valid_loader))

    print("Epoch: {}/{} \n".format(e + 1, epochs),
          "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
          "Validation Loss: {:.3f} ".format(val_loss / len(valid_loader)),
          "Training Accuracy: {:.3f} \n".format(train_accuracy/len(trainloader))
          )