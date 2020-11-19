import sys

try:
    import features.face_recognition.dataset_create as dc
    import features.face_recognition.train as train
    import features.face_recognition.predict as predict
except Exception as e:
    from auto_face_recognition.features.face_recognition import dataset_create as dc
    from auto_face_recognition.features.face_recognition import train as train
    from auto_face_recognition.features.face_recognition import predict as predict


class AutoFaceRecognition:
    def __init__(self):
        pass

    def datasetcreate(self, dataset_path='datasets', class_name='Demo',
                      haarcascade_path='haarcascade/haarcascade_frontalface_default.xml',
                      eyecascade_path='haarcascade/haarcascade_eye.xml', eye_detect=False,
                      save_face_only=True, no_of_samples=100,
                      width=128, height=128, color_mode=False):
        """
        Dataset Create by face detection
        :param dataset_path: str (example: 'folder_of_dataset')
        :param class_name: str (example: 'folder_of_dataset')
        :param haarcascade_path: str (example: 'haarcascade_frontalface_default.xml)
        :param eyecascade_path: str (example: 'haarcascade_eye.xml)
        :param eye_detect: bool (example:True)
        :param save_face_only: bool (example:True)
        :param no_of_samples: int (example: 100)
        :param width: int (example: 128)
        :param height: int (example: 128)
        :param color_mode: bool (example:False)
        :return: None
        """
        obj = dc.DatasetCreate(dataset_path=dataset_path, class_name=class_name,
                               haarcascade_path=haarcascade_path,
                               eyecascade_path=eyecascade_path, eye_detect=eye_detect,
                               save_face_only=save_face_only, no_of_samples=no_of_samples,
                               width=width, height=height, color_mode=color_mode)
        obj.create()

    def face_recognition_train(self, data_dir='datasets', batch_size=32, img_height=128, img_width=128, epochs=10,
                               model_path='model', pretrained=None, base_model_trainable=False):
        """
        Train TF Keras model according to dataset path
        :param data_dir: str (example: 'folder_of_dataset')
        :param batch_size: int (example:32)
        :param img_height: int (example:128)
        :param img_width: int (example:128)
        :param epochs: int (example:10)
        :param model_path: str (example: 'model')
        :param pretrained: str (example: None, 'VGG16', 'ResNet50', 'InceptionV3')
        :param base_model_trainable: bool (example: False (Enable if you want to train the pretrained model's layer))
        :return: None
        """

        obj = train.Classifier(data_dir=data_dir, batch_size=batch_size, img_height=img_height,
                               img_width=img_width, epochs=epochs, model_path=model_path, pretrained=pretrained,
                               base_model_trainable=base_model_trainable)
        obj.start()

    def predict_faces(self, class_name=None, img_height=128, img_width=128,
                      haarcascade_path='haarcascade/haarcascade_frontalface_default.xml',
                      eyecascade_path='haarcascade/haarcascade_eye.xml', model_path='model',
                      color_mode=False):
        """
        Predict Face
        :param class_name: Type-List (example: ['class1', 'class2'] )
        :param img_height: int (example:128)
        :param img_width: int (example:128)
        :param haarcascade_path: str (example: 'haarcascade_frontalface_default.xml)
        :param eyecascade_path: str (example: 'haarcascade_eye.xml)
        :param model_path: str (example: 'model')
        :param color_mode: bool (example: False)
        :return: None
        """
        obj = predict.Predict(class_name=class_name, img_height=img_height, img_width=img_width,
                              haarcascade_path=haarcascade_path,
                              eyecascade_path=eyecascade_path, model_path=model_path,
                              color_mode=color_mode)
        obj.cap_and_predict()

    def predict_face(self, class_name=None, img_height=128, img_width=128,
                     haarcascade_path='haarcascade/haarcascade_frontalface_default.xml',
                     eyecascade_path='haarcascade/haarcascade_eye.xml', model_path='model',
                     color_mode=False, image_path='tmp.png'):
        """
                Predict Face
                :param class_name: Type-List (example: ['class1', 'class2'] )
                :param img_height: int (example:128)
                :param img_width: int (example:128)
                :param haarcascade_path: str (example: 'haarcascade_frontalface_default.xml)
                :param eyecascade_path: str (example: 'haarcascade_eye.xml)
                :param model_path: str (example: 'model')
                :param color_mode: bool (example: False)
                :param image_path: str (example: 'src/image_predict.png'
                :return: None
                """
        obj = predict.Predict(class_name=class_name, img_height=img_height, img_width=img_width,
                              haarcascade_path=haarcascade_path,
                              eyecascade_path=eyecascade_path, model_path=model_path,
                              color_mode=color_mode, image_path=image_path)
        cls, confidence = obj.predict()
        return cls, confidence


if __name__ == '__main__':
    obj = AutoFaceRecognition()
    obj.datasetcreate()
