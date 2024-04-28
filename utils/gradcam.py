"""
Grad-CAM Implementation using Keras

Main links:
    - Grad-CAM Paper: https://arxiv.org/abs/1610.02391
    - Keras implementation: https://keras.io/examples/vision/grad_cam/
"""


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore


def read_image_to_array(img_path: str,
                        make_unit_batch: bool = True,
                        size=(128, 128)) -> np.ndarray:
    """
    Reads an image and returns it as an array.
    Optionally, the function can add a new dim.
    resembling a batch (needed for grad-cam).

    Args:
        img_path: Path to image.
        make_unit_batch: In case we want to add a new dimension.
                         Defaults to True.
        size: Image size. Defaults to (128, 128).

    Returns:
        Image as a numpy array.
    """
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    # normalizing pixel values [optional?]
    array /= 255.0
    
    if make_unit_batch:
        # adding a dimension to transform our array into a "batch"
        # of size  1, e.g. (1, 128, 128, 3)
        array = np.expand_dims(array, axis=0)
        
    return array 


class GradCAM:
    """
    Grad-CAM Implementation.
    
    Steps:
        1. Creates grad model.
        2. Given an image, creates the heatmap.
        3. Applies color map to heatmap.
        4. Overlays heatmap on top of image.
    """
    def __init__(self, model: tf.keras.models.Model,
                 last_conv_layer_name: str) -> None:
        self.grad_model = GradCAM.create_grad_model(model, last_conv_layer_name)
    
    @staticmethod
    def create_grad_model(model: tf.keras.models.Model,
                          last_conv_layer_name: str) -> tf.keras.models.Model:
        """
        Creates grad-cam model with 2 outputs, model's
        predictions and the output of the last conv
        layer.

        Args:
            model: keras vanilla model.
            last_conv_layer_name: name of last conv. layer.

        Returns:
            The model.
        """
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        return grad_model
    
    @staticmethod
    def create_heatmap(grad_model, batched_img_array, pred_index = None) -> np.ndarray:
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(batched_img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            
            try:
                class_channel = preds[:, pred_index]
            except:
                class_channel = tf.gather(preds, pred_index, axis=1)

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        return heatmap
    
    @staticmethod
    def apply_colormap_to_heatmap(heatmap, size=(128, 128)) -> np.ndarray:
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize(size)
        heatmap = np.array(heatmap)
        cmap_jet = plt.get_cmap('jet')
        # jet color map outputs 4 channels, we only want 3
        heatmap = cmap_jet(heatmap)[:, :, :3]  
        heatmap = heatmap
        return heatmap
    
    @staticmethod
    def overlay_image(img_array, resized_heatmap, alpha: float = 0.5) -> np.ndarray:
        if img_array.shape != resized_heatmap.shape:
            raise ValueError(f"Image ({img_array.shape}) and Heatmap ({resized_heatmap.shape}) shapes do not match.")
        
        overlayed_img = resized_heatmap * alpha + img_array
        return overlayed_img
    
    def get_superimposed_image(self,
                               batched_img_array,
                               pred_index = None,
                               alpha = 0.5):
                       
        heatmap = GradCAM.create_heatmap(self.grad_model, batched_img_array, pred_index=pred_index)
        heatmap = GradCAM.apply_colormap_to_heatmap(heatmap)
        # removing batch dim. (only needed to create heatmap)
        img_array = np.squeeze(batched_img_array)
        superimposed_img = GradCAM.overlay_image(img_array, heatmap, alpha=alpha)
        return superimposed_img
    

if __name__ == '__main__':
    # demonstration
    from tensorflow.keras.models import load_model  # type: ignore
    
    img_path = r'C:\Users\fmppo\Desktop\MSDSAA\Y1\S2\Deep Learning\deep-learning-23-24-nova-ims-group-project\data\train\pilar cyst\6c770bbf5b1394537ba9cb4c16050fe4.jpg'
    batched_img_array = read_image_to_array(img_path)

    model = load_model('model_dir/models/Xception-transfer-model.h5')
    gradcam = GradCAM(model=model, last_conv_layer_name='last_conv_layer')
    heatmap = GradCAM.create_heatmap(gradcam.grad_model, batched_img_array)  # calling this method for demo. purposes
    cmap_heatmap = GradCAM.apply_colormap_to_heatmap(heatmap)  # calling this method for demo. purposes
    superimposed_img = gradcam.get_superimposed_image(batched_img_array)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,6))

    ax1.imshow(np.squeeze(batched_img_array))
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.matshow(heatmap)
    ax2.set_title('Heatmap')
    ax2.axis('off')

    ax3.imshow(cmap_heatmap)
    ax3.set_title('Heatmap with color map')
    ax3.axis('off')

    ax4.imshow(superimposed_img)
    ax4.set_title('Super-imposed image')
    ax4.axis('off')

    plt.suptitle('Grad-CAM Implementation')
    plt.tight_layout()
    plt.subplots_adjust(top=1)
    plt.show()