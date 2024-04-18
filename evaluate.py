import torch
import torchvision
from torchvision.transforms import ToTensor, Lambda, Resize
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
from script import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model
verifier_model = ViTForImageClassification.from_pretrained("farleyknight/mnist-digit-classification-2022-09-04")
verifier_model = verifier_model.to(device)

# Ensure the model is in evaluation mode
verifier_model.eval()

n_classes = 10
n_feat = 128
n_T = 400

ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
            device=device, drop_prob=0.1)
ddpm = ddpm.to(device)

# optionally load a model
ddpm.load_state_dict(torch.load("./model_39.pth"))
ddpm.eval()

tf = torchvision.transforms.Compose([
    ToTensor(),
    Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel to get three channels
    Resize((224, 224))  # Resize the images to match the model's expected input size
])


def evaluate_accuracy(data_set):
    data_loader = DataLoader(data_set, batch_size=8, shuffle=True)
    total = 0
    correct = 0

    for batch in data_loader:
        images, labels = batch
        with torch.no_grad():
            outputs = verifier_model(images)

        # Get the predicted class for each image in the batch
        predicted_classes = torch.argmax(outputs.logits, dim=1)

        # Update the total and correct counts
        total += labels.size(0)
        labels = labels.to(device)
        correct += (predicted_classes == labels).sum().item()

    return correct / total


def evaluate_generation(model, n_sample, w, context):
    with torch.no_grad():
        x_gen, x_gen_store = model.sample(n_sample, (1, 28, 28), device, guide_w=w, context=context)

        # expected image size is 224x224
        x_gen = torch.nn.functional.interpolate(x_gen, size=(224, 224))
        # expand the channel dimension
        x_gen = x_gen.repeat(1, 3, 1, 1)

        c_gen = context.repeat(int(n_sample / context.shape[0]))
        c_gen = c_gen.to(device)

        data_set = TensorDataset(x_gen, c_gen)

        accuracy = evaluate_accuracy(data_set)

        return accuracy


def evaluate_generation_loop(model, w, context):
    with torch.no_grad():
        x_gen, x_gen_store = model.sample(0, (1, 28, 28), device, guide_w=w, context=context)

        x_gen_store = x_gen_store.squeeze(1)

        x_gen_store = torch.nn.functional.interpolate(x_gen_store, size=(224, 224))
        x_gen_store = x_gen_store.repeat(1, 3, 1, 1)

        context = context.repeat(x_gen_store.shape[0])

        data_set = TensorDataset(x_gen, context)

        accuracy = evaluate_accuracy(data_set)

        return accuracy

def load_mnist():
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=tf, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=tf, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def generate_and_show_10(model, w, context):
    with torch.no_grad():
        x_gen, x_gen_store = model.sample(10, (1, 28, 28), device, guide_w=w, context=context)

        # plot the generated images
        plt.figure(figsize=(5, 2))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_gen[i, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
        plt.show()


def generate_and_show_all_digits(model, w):
    for i in range(0, 10):
        context = torch.tensor([i])
        generate_and_show_10(model, w, context)


def performance_by_context():
    accuracies = []
    n_samples = 400

    for i in range(0, 10):
        acc = evaluate_generation(ddpm, n_samples, 0.0, torch.tensor([i]))
        print(f"Accuracy for context digit {i}: {acc}")
        accuracies.append(acc)

    # plt accuracy against the context i.e. different digits
    fig, ax = plt.subplots()

    ax.bar(range(0, 10), accuracies)
    ax.set_xlabel("Context Digit")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of Generated Images Against Context Digit")
    plt.show()


def performance_by_weight_on_9():
    accuracies = []
    n_samples = 400

    for i in range(0, 10):
        acc = evaluate_generation(ddpm, n_samples, i / 10, torch.tensor([9]))
        print(f"Accuracy for weight {i / 10} on context digit 9: {acc}")
        accuracies.append(acc)

    # plt accuracy against the weight
    plt.figure()
    plt.plot([i / 10 for i in range(0, 10)], accuracies)
    plt.xlabel("Weight")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Generated Images Against Weight on Context Digit 9")
    plt.show()


def performance_by_training_iterations():
    accuracies = []
    n_samples = 100

    for i in range(0, 10):
        ddpm.load_state_dict(torch.load(f"./data/diffusion_outputs/model_{i}.pth"))
        acc = evaluate_generation(ddpm, n_samples, 0.0, torch.arange(10))
        print(f"Accuracy for model trained for {i} iterations: {acc}")
        accuracies.append(acc)

    # plt accuracy against the training iterations
    plt.figure()
    plt.plot(range(0, 10), accuracies)
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Generated Images Against Training Iterations")
    plt.show()


# def performance_by_sampling_iterations():
#     for i in range(5):
#         evaluate_generation_loop(ddpm, 0.0, torch.tensor([9])


if __name__ == '__main__':
    performance_by_weight_on_9()

# if __name__ == '__main__':
#     n_samples = 40
#     # Load the MNIST dataset
#     mnist = torchvision.datasets.MNIST(root='.', download=True, transform=tf)
#     mnist = torch.utils.data.Subset(mnist, range(n_samples))
#     # move to device
#     mnist = [(x.to(device), y) for x, y in mnist]
#     accuracy = evaluate_accuracy(mnist)
#     print(f"Accuracy: {accuracy * 100:.2f}%")
