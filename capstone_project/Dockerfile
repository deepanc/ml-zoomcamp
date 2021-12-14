FROM agrigorev/zoomcamp-cats-dogs-lambda:v2

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl?raw=true
RUN pip install requests

COPY xception-clothing-model.tflite .
COPY lambda_function.py .
COPY dataset/test_black.jpg .
COPY dataset/test_red.jpg .
COPY dataset/test_yellow.jpg .

CMD ["lambda_function.lambda_handler"]