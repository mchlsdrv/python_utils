import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def get_x_ticks(epoch):
    x_ticks = np.arange(1, epoch)
    if 20 < epoch < 50:
        x_ticks = np.arange(1, epoch, 5)
    elif 50 < epoch < 100:
        x_ticks = np.arange(1, epoch, 10)
    elif 100 < epoch < 1000:
        x_ticks = np.arange(1, epoch, 50)
    elif 1000 < epoch < 10000:
        x_ticks = np.arange(1, epoch, 500)

    return x_ticks


def plot_loss(train_losses, val_losses, x_ticks: np.ndarray, x_label: str, y_label: str,
              title='Train vs Validation Plot',
              train_loss_marker='bo-', val_loss_marker='r*-',
              train_loss_label='train', val_loss_label='val', output_dir: pathlib.Path or str = './outputs'):
    fig, ax = plt.subplots()
    ax.plot(x_ticks, train_losses, train_loss_marker, label=train_loss_label)
    ax.plot(x_ticks, val_losses, val_loss_marker, label=val_loss_label)
    ax.set(xlabel=x_label, ylabel=y_label, xticks=x_ticks)
    fig.suptitle(title)
    plt.legend()
    output_dir = pathlib.Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_output_dir = output_dir / 'plots'
    os.makedirs(plot_output_dir, exist_ok=True)
    fig.savefig(plot_output_dir / 'loss.png')
    plt.close(fig)
