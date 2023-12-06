def prepare_image(image_path):
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = keras.applications.mobilenet.preprocess_input(img_array)
            return img_array