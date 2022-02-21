# People-Counter-In-ROIs

### Output Sample:
[(https://github.com/Jabalov/People-Counter-And-Detection-In-Given-ROIs/blob/main/output1.avi)]


## Overview:
This task is pretty simple, it's about detecting people and counting them in given ROIs (regions-of-interest) in video frames.
Using c++ opencv to read the model configuration (YOLO-Fastest), do the forward pass and the output postprocessing (boxes resized, filtered classes to people only, counting objects in each roi and more).
