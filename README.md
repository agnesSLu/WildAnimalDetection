# Wild Animal Detection

The goal of this project is to develop a model for wild animal detection, allowing real time detection and video upload.

Including 10 species:
buffalo, elephant, rhino, zebra, cheetah, lion, tiger, fox, wolf, hyena

![Interface](interface.png)

## Team member
Chuhan Ren

Shihua Lu


## Link to presentation and demo

https://drive.google.com/drive/u/0/folders/1t92qH4GkPS0DCvu4kTjaIQmelPtetGgl

## Dataset
1. African Wildlife (manually input YOLO format): https://www.kaggle.com/datasets/biancaferreira/african-wildlife 

2. Wild animals: https://www.kaggle.com/datasets/vishweshsalodkar/wild-animals 

3. Wild animal images: https://www.kaggle.com/datasets/whenamancodes/wild-animals-images

## Getting Started

1. Fork the repo

2. Clone the forked repository

3. Install dependencies

```
pip install gradio ultralytics opencv-python
```

4. Upload video to detect wild animals (slow):

```
cd app
python app.py
```

server run on http://127.0.0.1:7860

5. Real time detection run locally:

```
cd app
python camera.py
```

6. For faster detection using GPU and two implementations:

For now can access:
https://4538bd0a6383279858.gradio.live/

Will deploy to Hugging Face in the future


