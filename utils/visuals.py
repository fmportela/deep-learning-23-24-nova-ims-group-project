"""
Storing commonly used plotting functions
used throughout the notebooks
"""


from typing import Union
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from .constants import CLASS_INT_TO_STR


class SampleVisuals:
    """
    Class for storing methods pertaining to showing
    samples from image datasets.
    """
    
    @staticmethod
    def show_random_sample(data: tf.data.Dataset,
                           batch_size: int,
                           sup_title: str = None,
                           label_mode: str = 'categorical',
                           grid_dim: tuple = (3, 3)) -> None:
        """
        Shows a random sample of images.

        Args:
            data: Dataset.
            label_mode: Class names depend on the 'label_mode'
                        of the import. Defaults to 'categorical'.
            grid_dim: Grid dimensions .Defaults to (3, 3).
        """
        height, width = grid_dim
        n = height * width
        class_names = list(CLASS_INT_TO_STR.values())
        data = data.rebatch(n)
        ds_len = sum(1 for _ in iter(data))
        skips = np.random.choice(range(ds_len))
        plt.figure(figsize=(10, 10))
        if sup_title:
            plt.suptitle(sup_title)
        for images, labels in data.skip(skips).take(n):
            for i in range(n):
                ax = plt.subplot(height, width, i + 1)
                ax.imshow(images[i])
                if label_mode == 'int':
                    ax.set_title(class_names[labels[i]])
                elif label_mode == 'categorical':
                    ax.set_title(class_names[np.where(labels[i]==1.0)[0][0]])
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_class_sample(data: tf.data.Dataset,
                          class_name: str,
                          label_mode: int = 'int',
                          grid_dim: tuple = (3, 3)) -> None:
        """
        Shows a sample from a specific class.

        Args:
            data (tf.data.Dataset): Dataset.
            class_name (str): Name of class to sample from.
            label_mode (int, optional): Class names depend on the 'label_mode'
                                        of the import. Defaults to 'categorical'.
            grid_dim (tuple, optional): Grid dimensions. Defaults to (3, 3).

        Raises:
            NotImplementedError: If label mode is 'int'
        """

        height, width = grid_dim
        n = height * width
        
        # class index
        class_idx = np.where(np.array(data.class_names) == class_name)[0][0]
        
        # iterating over data to fill as many
        # samples of the same class as needed
        iter_data = iter(data)
        grid_images = []
        while len(grid_images) < n:
            curr = next(iter_data)
            for i, label in enumerate(curr[-1]):
                if label_mode == 'int':
                    raise NotImplementedError('TODO')
                elif label_mode == 'categorical':
                    if np.where(label == 1.0)[0][0] == class_idx:
                        grid_images.append(curr[0][i])
        
        plt.figure(figsize=(10, 10))
        plt.suptitle(f'Class: {class_name}')
        for i in range(n):
            plt.subplot(height, width, i + 1)
            plt.imshow(grid_images[i].numpy().astype("uint8"))
            plt.axis("off")


class ModelVisualEvaluation:
    
    @staticmethod
    def plot_history_f1(history: dict,
                        figsize: tuple = (8, 4)) -> None:
        """
        Plots train and validation loss and F1 Score.
        
        Args:
            history: history dictionary returned from
                    tf.keras.model.fit method.
            figsize: figure size. Defaults to (8, 4).
        
        Returns:
            None, it displays and image.
        """
        f1 = history['f1_score']
        val_f1 = history['val_f1_score']
        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(1, len(f1) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(epochs, f1, 'bo', label='Training F1')
        ax1.plot(epochs, val_f1, 'b', label='Validation F1')
        ax1.set_title('Training and validation F1 Score')
        ax1.legend()

        ax2.plot(epochs, loss, 'bo', label='Training loss')
        ax2.plot(epochs, val_loss, 'b', label='validation loss')
        ax2.set_title('Training and validation loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_prediction_sample(X: np.ndarray,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               show_class: Union[int, None] = None,
                               only_show_wrong: bool = True,
                               grid_size: tuple = (3, 3),
                               class_names: dict = CLASS_INT_TO_STR):
        """
        This function plots the images alongside
        their predicted and true classes. We can
        opt for plotting random incorrect instances
        or correct ones. We can also choose to plot
        instances belonging to one specific class (true
        class).
        
        Args:
            X: data to recreate image
            y_true: true class
            y_pred: predicted class
            show_class: in case we want to only plot
            a specific class
            only_show_wrong: if True, show only incorrectly
            predicted images, else shows correct instances
        
        Returns:
            None
        """
             
        height, width = grid_size
        n = height * width  # number of subplots
        
        # from which set of instances can we choose from
        # np.where returns the positions of the instances we want    
        if only_show_wrong:
            if show_class:
                possibilities = np.where((y_true != y_pred) & (y_true == show_class))[0]
            else:
                possibilities = np.where(y_true != y_pred)[0]
        else:
            if show_class:
                possibilities = np.where((y_true == y_pred) & (y_true == show_class))[0]
            else:
                possibilities = np.where(y_true == y_pred)[0]
        
        # need to fill the whole grid
        if len(possibilities) < n:
            raise ValueError(f'No. of possibilities: {len(possibilities)} < grid size: {n}.') 
            
        # choosing randomly (based on indices)
        chosen_idx = np.random.choice(possibilities, n, replace=False)
        
        # iterating over data to get images at those indices 
        images = []
        for i, img in enumerate(X):
            if i in chosen_idx:
                # (image, true class, predicted class)
               images.append((img,
                              class_names[y_true[i]],
                              class_names[y_pred[i]])) 
               
        # creating plot
        plt.figure(figsize=(10, 10))
        for i in range(n):
            plt.subplot(height, width, i + 1)
            plt.imshow(images[i][0])
            plt.title(f'Predicted  Class: {images[i][2]}\nTrue class: {images[i][1]}', fontsize=8)
            plt.axis("off")
        
        if only_show_wrong:
            plt.suptitle('Random sample of INCORRECT classifications')
        else:
            plt.suptitle('Random sample of CORRECT classifications')


if __name__ == '__main__':
    #test
    pass