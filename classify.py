from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


# load and prepare the image
def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    # Image array of pixels & colours
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    # Numbers pixels with values
    img = img.reshape(1, 32, 32, 3)

    img = img.astype('float32')
    # Normalises values
    img = img / 255.0
    return img


def run_example():
    img = load_image('sample_image-1.png')
    model = load_model('final_model.h5')
    result = model.predict_classes(img)
    print(result[0])


run_example()
