# Which Star Is Your Lookalike?

This is a repo with a demo of the last project I did while in Polytech Marseille.
This project was done in collaboration with my colleague Fernando Callera.

The demo is a simple Streamlit web page, which should be enough to give you a overview of the system and all the code you'd need should you try to replicate it. The actual project consists of a FastAPI backend with a Vue.js frontend.

First of all, you need some models:

- dlib_face_recognition_resnet_model_v1.dat
- shape_predictor_5_face_landmarks.dat

Available at https://github.com/davisking

- GFPGANv1.4.pth

Available at https://github.com/TencentARC/GFPGAN

- inswapper_128.onnx

Not available publicly by the creators. I also won't upload it myself in respect to their wishes. If you do find it, don't use it for commercial or unethical purposes.

Now for the dataset, I've made my own since this project requires very clear portrait pictures. You can just download peoples faces on the internet and put their name as the filename. image_standardizer.py can be used for making all the pictures .jpg!

```
python image_standardizer.py .\faces\ .\standardized\  
```


If you don't want to worry with this, here's my dataset:

https://drive.google.com/drive/folders/1Z_LzvDSZzA3B1ZWf8YYA24Ax7pkE2sTg?usp=sharing

Do not use it for commercial purposes, the images are not for public use. If you want this file taken down, please contact copyright@dielson.com.br.

Finally, you have to populate the database with the faces. Be sure to put the right paths for the models and the portraits.

```
python celebrity_db_populator.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ./standardized
```


Finally, running the application:
```
streamlit run celebrity_lookalike_app.py
```

