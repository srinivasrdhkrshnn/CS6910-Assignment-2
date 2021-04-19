# CS6910-Assignment-2: Convolutional Neural Networks

### Steps to run the program
- The code is done in a Google colab notebook. It can be opened and run in Google colab or jupyter notebook.
- Each question is made in seperate code cells. 
- Run the whole notebook to get the outputs for all questions
- The sweep for hyperparameter sweeps is commented out by default. Uncomment the cell to run the sweep across the given dictionary

## Part A
To train the cnn, use /partA/Assignment_2.ipynb
To train a custom convolutional neural network, edit the values in the following dictionary

```python
config_defaults = {
    "img_size" : 229,
    "batch_size" : 100,
    "n_classes" : 10,
    "n_filters" : [16,32,64,64,128],
    "filter_size" : [3,3,3,3,3],
    "fc_size" : 400,
    "drop_out" : 0.5,
    "augmentation" : 1,
    "batch_normalize" : 1 
}
``` 

  where :
  ```
  img_size -- (int) Dimension of image
  batch_size -- (int) Batch size
  n_classes -- (int) Number of output classes
  n_filters -- (list) Number of filters in each layer
  filter_size -- (list) Size pf filters in each layer
  fc_size -- (int) Fully connected layer size
  drop_out -- (float) Dropout percentage (0-1)
  augmentation -- (int) 0/1 to disable/ enable data augmentation
  batch_normalize -- (int) 0/1 to disable/ enable batch normalization
  ```

Then, run 
```python
train()
```
The resulting model will be stored in wandb.

To run a sweep, run

```python
sweeper(sweep_config,PROJECT_NAME)
```

where :
  ```
  sweep_config -- (dict) wandb sweep config dictionary
  PROJECT_NAME -- (string) wandb project name
  
  ```
  
For the remaining questions in PartA, run /partA/Assignment_2a45.ipynb

We use the best model obtained from wandb for all results

Running the full code gives 
- a 3 x 10 grid of random images with their predictions.
- a visualisation of first layer filter outputs for a random image
- Guided backpropogation visualisation 

All outputs are logged in wandb

To obtain Guided backpropogation visualisation, run 
```python
guidbp(image)
```

where :
  ```
  image -- (numpy array) image input for guided backpropogation (229 x 229)
  ```
  
## Part B

Use Assignment_2_B.ipynb

To train a custom convolutional neural network, edit the values in the following dictionary

```python
config_defaults = {
  "model_name" : "InceptionV3",
  "fc_size" : 1024,
  "augmentation" : 1,
}
```

where :
  ```
  model_name -- (string) pretrained model name- ['InceptionV3','MobileNetV2','InceptionResNetV2','ResNet50','Xception']
  fc_size -- (int) Fully connected layer size
  augmentation -- (int) 0/1 to disable/ enable data augmentation
  
  ```
  
Then, run 
```python
train()
```
The resulting model will be stored in wandb.

To run a sweep, run

```python
sweeper(sweep_config,PROJECT_NAME)
```

where :
  ```
  sweep_config -- (dict) wandb sweep config dictionary
  PROJECT_NAME -- (string) wandb project name
  
  ```
Link for WandB report: Part A: https://wandb.ai/cs6910krsrd/CS6910%20ASSIGNMENT%202/reports/CS6910-Assignment-2-Part-A---Vmlldzo2MTUwOTE
                       Part B & C: https://wandb.ai/krsrinivas/CS6910%20ASSIGNMENT%202B/reports/CS6910-Assignment-2-Part-B-C---Vmlldzo2MTA4ODE
