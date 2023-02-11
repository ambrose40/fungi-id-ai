# fungi-id-ai
Fungi ID AI recognition by photo

## Keywords:
keras, numpy, tensorflow, pyplot, opencv,

dataset, h5, gpu, model,

fungi, photo, id, recognition, ai.


## Datasets:

### 33 fungi species with photos count in each folder >=40 (resized to 256x256) - 1971 images total:


https://drive.google.com/file/d/1EK_pZeSHQkDMUV4sy918tffr-G9mZQC0/view - .zip

https://drive.google.com/drive/folders/1pIJznjmFhs3n6Gsvre60H4WobovwVgRB - folder 

### 496 fungi species with no photos count filter (>=1) (resized to 128x128) - 7572 images total:

https://drive.google.com/file/d/1aOYh0IMxQgXNihY7GARD3cp_L6VdpA_8/view - .zip

https://drive.google.com/drive/folders/1YnymU79AR4q0i7Ify9kPUt_cWZf81t0k - folder 

## Pre-trained model (h5):

### Model summary:
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 sequential (Sequential)     (None, 256, 256, 3)       0         
                                                                 
 rescaling (Rescaling)       (None, 256, 256, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 256, 256, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 128, 128, 32)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 128, 64)      18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 64, 64, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 64, 64, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 32, 32, 128)      0         
 2D)                                                             
                                                                 
 dropout (Dropout)           (None, 32, 32, 128)       0         
                                                                 
 flatten (Flatten)           (None, 131072)            0         
                                                                 
 dense (Dense)               (None, 256)               33554688  
                                                                 
 outputs (Dense)             (None, 33)                8481      
                                                                 
=================================================================
Total params: 33,656,417
Trainable params: 33,656,417
Non-trainable params: 0
_________________________________________________________________
```

### Download:

https://drive.google.com/file/d/11sytuJvpxeyk663KzsCpg6rpZLX-aHp7/view - .zip

https://drive.google.com/drive/folders/1Ge7lgXrQ6pIHGrWPa1lpBKgtQJuetk-Y - folder 
