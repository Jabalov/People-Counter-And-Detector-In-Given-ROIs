# People-Counter-In-ROIs

### Output Sample:
https://user-images.githubusercontent.com/83673888/155828087-26896d8a-3735-47d1-bdbe-a1060b076646.mp4


## Overview:
This task is pretty simple, it's about detecting people and counting them in given ROIs (regions-of-interest) in video frames.
Using c++ opencv to read the model configuration (YOLO-Fastest), do the forward pass and the output postprocessing (boxes resized, filtered classes to people only, counting objects in each roi and more).
