from rest_framework.parsers import FileUploadParser, MultiPartParser
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import UploadSerializer
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import action
from drf_yasg import openapi
from rest_framework.status import HTTP_400_BAD_REQUEST
from keras.preprocessing import image
import numpy as np
import keras
import os
from unidecode import unidecode
import json
import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator



def find_json_by_name(file_path, vname):
    # Đọc dữ liệu từ file JSON
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Chuyển đổi các phần tử trong tập hợp vname thành văn bản không dấu
    vnames_normalized = {unidecode(name) for name in vname}

    # Tìm kiếm trong danh sách JSON
    matching_items = []
    for item in data:
        # Đảm bảo rằng cả 'item.get('name')' là chuỗi
        item_name = str(item.get('name', ''))
        if unidecode(item_name) in vnames_normalized:
            matching_items.append(item)

    # Trả về danh sách các mục phù hợp, hoặc None nếu không có mục nào phù hợp
    return matching_items or None


json_path="C:\\Users\\phuc2\\Downloads\\ai-pbl6\\pbl6\\uploadimage\\data\\data.json"

class UploadFile(APIView):
    
    parser_classes = [MultiPartParser]
    @swagger_auto_schema(
        operation_description='Upload Image...',
        manual_parameters=[
            openapi.Parameter('image', openapi.IN_FORM, type=openapi.TYPE_FILE, description='Image to be uploaded'),
        ]
    )
    @action(detail=True, methods=['POST'])
    def post(self, request):
        serializer = UploadSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"errors": serializer.errors},
                status=HTTP_400_BAD_REQUEST
            )

        instance = serializer.save()

        model = load_model("C:\\Users\\phuc2\\Downloads\\ai-pbl6\\pbl6\\uploadimage\\model.h5")

        def prepare_image(image_path):
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = keras.applications.mobilenet.preprocess_input(img_array)
            return img_array

        data_folder = "C:\\Users\\phuc2\\Downloads\\ai-pbl6\\pbl6\\uploadimage\\data"
        train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                           rotation_range=0.2,
                                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.3, zoom_range=0.5,
                                           horizontal_flip=True, vertical_flip=True,
                                           validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(data_folder,
                                                            target_size=(224, 224),
                                                            batch_size=64,
                                                            class_mode='categorical',
                                                            subset='training')

        img_array = prepare_image(str(instance.image))

        validation_generator = train_datagen.flow_from_directory(
            data_folder,
            target_size=(224, 224),
            batch_size=64,
            class_mode='categorical',
            subset='validation'
        )

        classes = train_generator.class_indices
        classes = list(classes.keys())

        # Dự đoán
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        result = {classes[predicted_class]}
        result_json= find_json_by_name(json_path, result)
        # Xóa ảnh sau khi nhận diện
    
        os.remove(str(instance.image))

        return Response(result_json)
