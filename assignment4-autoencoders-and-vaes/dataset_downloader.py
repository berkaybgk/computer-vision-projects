import torchvision.datasets as datasets


if __name__ == '__main__':
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    print("Number of training samples:", len(mnist_train))
    print("Number of test samples:", len(mnist_test))

    print("MNIST dataset downloaded successfully.")
