# NeuroStyle

This repository hopes to recreate the style transfer effect of other popular neural nets (made mostly to practice with keras). 
Currently, it only creates grayscale images.

An example of an image generated from the "marker.h5" model created via NeuroStyle:
![alt text](https://github.com/jlehett/NeuroStyle/blob/master/saved_images/reach_sketch.PNG "Example 1")

## Getting Started

First, add 3 empty folders to the root directory: "training", "truth", "unused". Since these folders were empty, they did not get pushed to the repo.

To run NeuroStyle, simply run ui.py. This will pull up a Tkinter window that allows for training of models, viewing of models, and adjustment of settings.

### Prerequisites

Utilizes os module for saving files, etc. Might not work on systems other than Windows.

Libraries required:
  1. Keras
  2. Numpy
  3. Tkinter
  4. cv2
  
## The GUI

The Tkinter window that is opened upon running ui.py is split into a top half which is for training models, and a bottom half which is for viewing models. 
 
### Training Models

To train a model, you first need to select a model to train. You can press the Model button to open up a filedialog and select a .h5 file to use as a model. If you want to start training a new model, or if you don't have any models to use already, press the New button next to Model Options. This will open a popup window asking for you to enter a name for your new model. Press Submit to finalize and open up your model with the Model button.

Once you have a model selected, it is time to add images to train on. Press the Add button next to Training Pictures to bring up a filedialog. First, select a .png or .jpg file to use as a base image. Then another filedialog will pop up. Select an image to pair with the base image. Use the base image with whatever filter you want to have the AI learn applied to it as this second image (make sure they are the same image, but have a filter applied to one, and make sure the images are the same size).

If you would like to remove an image pair from the training pool, select the Delete button and select the base image you would like to remove from the pool. This will remove both the base image and the image it is paired to.

Before training, look over the Training Settings. Epochs defines the number of iterations to train for. Batch size defines number of samples to analyze before adjusting weights. Pixels Per Image defines the number of pixels to randomly select from each image in the training pool to use in training. Subsection Radius defines the radius around the pixel to be used as inputs to the neural net to predict the center pixel. If a pixel does not exist (for example, if the selected pixel is in the upper left hand corner) it defaults to 0.

When you are ready, press the Train button to begin training. Currently, you will only see progress through the console you are running ui.py in. You can save a copy of the current model to a new file by pressing the Save As button.

### Viewing Models

To view the results of a model, you first need to select a model to use in predictions. This is found under the Creation Settings. You can also choose the output image size in pixels. Currently, the program only supports square images. If, for example, you choose 600 as your Image Output Size, the resulting images will be of size 600x600 pixels.

You can add images to the queue of images to produce under the To Create list. Press the Add button under the list and choose a .png or .jpg picture to add to the queue. If you would like to delete an image from the queue, select it from the list and press the Delete button. If no image is selected, Delete will delete the most recently added image from the queue.

There is a check box under the To Create queue that reads "View on create?" If this is checked, after every image is produced, it will be displayed to the user's screen, and the user must press any key to close the window again before the program will move on to the next image in the queue. All images produced will be placed into the "saved_images" folder in the root directory of the program. They will be named the same as the file placed into the queue. They are saved as .png's with 0 as the compression level.

Once all settings are finalized, press the Start Creation Queue button to begin. Once again, all progress is currently displayed only in the console that is running ui.py.

## The Neural Net

The neural net in this program is trained on numpy arrays where the center of the array is the pixel the system is trying to predict, and the rest of the array is the square radius of pixels around the center pixel defined by Subsection Radius in the Training Settings. When predicting an image, the system runs pixel by pixel to make predictions for what that value that pixel would hold in the new image. (Yes, this is slow, and I tried doing a CNN at first, but this seemed far easier to learn and start with. I am hoping to improve the neural net in the future.)

## Improvements to Make

As I explained in the above section, the neural net currently implemented takes a long time to produce an output image since it is reading pixel by pixel. I am hoping to switch to a CNN or something more effective in future updates. I also was unsure of the best model to use. Currently, it is just made up of 3 dense layers and uses 'relu' activation functions. 

The GUI is also, admittedly, not the greatest. I'm not very familiar with Tkinter, but I wanted to have a GUI for this program to make training models far faster and easier. Though it is far down on my list of priorities, improvements to the GUI would be great.

Overall code organization could use a little work. Especially for GUI code, since as I said, I was unfamiliar with Tkinter which resulted in poor code structure. Switching over to a GUI also forced me to make a few adjustments to other .py files which could be cleaned up as well.

## Examples

Examples of one neural network I trained can be found in the saved_images folder already. I trained this neural network on 2 images, one of an apple, and one of an orange, which I paired with hand drawn outlines of the most defining features. This resulted in a cool marker or charcoal-drawn look. If you would like to use this model on your own images, the model is stored in the "models" folder under the name "marker.h5"

## Note on "unused" Folder

This folder is where you can store images that you want to train on or filter with a model. Not necessary, but helpful to keep all images in one place.
