import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 绘制acc及loss曲线

    acc = np.random.rand(200)
    val_acc = np.random.rand(200)

    loss = np.random.rand(200)
    val_loss = np.random.rand(200)

    epochs_range = range(200)

    plt.figure(figsize=(16, 16))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('1')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('2')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('3')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('4')

    plt.show()
    # img = plt.gcf()
    # img.savefig("./data/img.png")
    # img.clear()