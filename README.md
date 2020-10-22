# [auto_face_recognition](https://github.com/Dipeshpal/auto_face_recognition)

***Last Upadted: 02 September, 2020***

1. What is auto_face_recognition?
 2. Prerequisite
 3. Getting Started- How to use it?
 4. Future?

## 1. What is auto_face_recognition?
It is a python library for the Face Recognition. This library make face recognition easy and simple. This library uses Tensorflow 2.0+ for the face recognition and model training.

## 2. Prerequisite-

* To use it only Python (> 3.6) is required.

## 3. Getting Started (How to use it)-
 
 ### Install the latest version-
 `pip install auto_face_recognition`

It will install all the required package automatically, including Tensorflow Latest.


### Usage and Features-

After installing the library you can import the module-

1. **Object Creation-**
	```
	import auto_face_recognition
	obj = auto_face_recognition.AutoFaceRecognition()
	```
2. **Dataset Creation-**
 

	    obj.datasetcreate(haarcascade_path='haarcascade/haarcascade_frontalface_default.xml',  
	    	    	                  eyecascade_path='haarcascade/haarcascade_eye.xml') 
                  
	***Note:*** You need to pass the '[haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)' and '[haarcascade_eye.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)' path.

3. **Model Training-**

		obj.face_recognition_train()		

4. **Predict Faces-**

	    # Real Time
	    obj.predict_faces()
	    # Single Face Recofnition
	    obj.predict_face()

**Parameters You Can Choose-**

datasetcreate

    datasetcreate(dataset_path='datasets', class_name='Demo',  
                      haarcascade_path='haarcascade/haarcascade_frontalface_default.xml',  
                      eyecascade_path='haarcascade/haarcascade_eye.xml', eye_detect=False,  
                      save_face_only=True, no_of_samples=100,  
                      width=128, height=128, color_mode=False)
    """"                  
	Dataset Create by face detection  
	:param dataset_path: str (example: 'folder_of_dataset')
	:param class_name: str (example: 'folder_of_dataset')
	:param haarcascade_path: str (example: 'haarcascade_frontalface_default.xml)
	:param eyecascade_path: str (example: 'haarcascade_eye.xml):param eye_detect: bool (example:True)
	:param save_face_only: bool (example:True)
	:param no_of_samples: int (example: 100)
	:param width: int (example: 128)
	:param height: int (example: 128)
	:param color_mode: bool (example:False):return: None
	"""  
face_recognition_train

    face_recognition_train(data_dir='datasets', batch_size=32, img_height=128, img_width=128, epochs=10,  
                               model_path='model'):  
     """  
     Train TF Keras model according to dataset path  
     :param data_dir: str (example: 'folder_of_dataset')  
     :param batch_size: int (example:32)  
     :param img_height: int (example:128)  
     :param img_width: int (example:128)  
     :param epochs: int (example:10)  
     :param model_path: str (example: 'model')  
     :return: None  
     """
                   
   predict_faces
       
    predict_faces(self, class_name=None, img_height=128, img_width=128,  
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

predict_face
	

    predict_face(self, class_name=None, img_height=128, img_width=128,  
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

## 4. Future?

	Finetuning with Resnet and others.
	You Suggest.
	
### Like my work?

Start the project and subscribe me on [YouTube](https://www.youtube.com/dipeshpal17).
https://www.youtube.com/dipeshpal17
