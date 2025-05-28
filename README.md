# fungi-id-ai
Fungi ID AI recognition by photo

## Keywords:
keras, numpy, tensorflow, pyplot, opencv,

dataset, h5, gpu, model, python

fungi, photo, id, recognition, ai.


## Datasets:

### 33 fungi species with photos count in each folder >=40 (resized to 256x256) - 1971 images total:

https://drive.google.com/file/d/1bBfDrTErmaiZyXysZLHhV9KYam8FKtaj/view - .zip

https://drive.google.com/drive/folders/1oYy6H3CRx6N89HNRZhPOaXEjubI2T1eF - folder 

### 496 fungi species with no photos count filter (>=1) (resized to 128x128) - 7572 images total:

https://drive.google.com/file/d/1vZshtPGad_QPTNx2YrTecdCYKQj-dxnD/view - .zip

https://drive.google.com/drive/folders/1ya4f5K4DLJNsD-fGG0tNyisZNEk7ge4K - folder 

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

https://drive.google.com/file/d/1Lg7xGaeXeRX4IZAVriDyXwWcf65aGf3w/view - .zip

https://drive.google.com/drive/folders/1axdzaJgrHk4E7d1ccuI-LtY9KywO7MHx - folder 
