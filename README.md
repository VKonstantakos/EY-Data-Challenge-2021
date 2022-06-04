
# EY Better Working World Data Challenge 2021

Global semi-finalist solution to the EY Better Working World Data Challenge 2021.


  ## Table of contents

* [Introduction](#Introduction)
* [Training Process](#Training_Process)
* [Figures](#Figures)
* [Results](#Results)
* [Support](#Support)

## Introduction

Active fire detection from airborne imagery is of critical importance to the management of environmental conservation policies, supporting decision-making and law enforcement. This fact becomes
especially evident during bushfire season, when accurate and quick information about the location
and rate of spread of active fires is required. To collect this information, aircraft carrying infrared
cameras fly over and record the intensity and location of fires. The resulting images are known as infrared linescans and are currently considered one of the best sources of information about fire intensity and location. After acquiring these images, the fire boundaries in each linescan image are
manually labeled by hand-drawing polygons around the edges of the fire using geospatial software.
However, during times of intense and rapid fire activity, this process can create a bottleneck in
delivering timely information to operational firefighting teams. For this reason, we were motivated
to propose and implement a system that performs this mapping task automatically, allowing for a
more effective allocation of human resources.

## Training Process

### Pre-processing

To begin with, we loaded and prepared the given dataset into a suitable format for the following
steps of the process. After the initial data exploration, it became evident that we needed to
clean up the noise in the images and make their signal clearer. This was especially important in
order to extract meaningful information and make model training easier and more
robust. To accomplish that, we explored some traditional approaches in image segmentation, such
as thresholding, edge detection and clustering (see [Figures](#Figures)).

- Thresholding: automatically determines the optimal threshold according to acertain criterion, and uses this cut-off. This threshold is determined based on the properties of the whole image or the neighboring pixels alone, defining a global and local thresholding method, respectively.
    - Manual thresholding.
    - Adaptive thresholding.
    - Yen's method.
- Edge detection: uses the different characteristics (e.g., local brightness, color discontinuity) of the important regions to achieve image segmentation. This discontinuity can often be detected using derivative operations and differential operators.
    - First-order Sobel operator.
    - Second-order Laplacian operator (Best one based on our evaluation).
- Clustering: uses the similarity between things as the criterion of class division, so that the same kind of samples are as similar as possible, and different ones are as dissimilar as possible.
    - Simple Linear Iterative Clustering (SLIC).
    - Quick shift algorithm.

### Model development

 - Apply Laplacian of Gaussian (LoG) operator.
 - Resize images into (768, 768).
 - Experiment with several neural network architectures (see [Figures](#Figures)).
     - U-Net.
     - LinkNet.
     - Feature Pyramid Networks (FPN).
     - Pyramid Scene Parsing Network (PSPNet).
     - All models used the VGG16 architecture for the encoder part.
 - Image augmentation with Keras Image Generator: horizontal/vertical flip, rotation, height/width shift, and brightness shift.
 - Loss functions:  Binary Cross-Entropy, Jaccard Loss, Dice Loss, Tversky Loss, Focal Loss.
 - Optimizers: Adam, RMSProp, Stochastic Gradient Descent.
 - Evaluation metric: F<sub>1</sub> score.
 - The chosen model was trained for 50 epochs with a batch size of 2, and the best model according to the validation score was kept (FPN with Jaccard loss and Adam optimizer).

### Post-processing

Post-processing methods refine a found segmentation and remove obvious errors. For example, the
morphological operations opening and closing can remove noise. The opening operation is a dilation
followed by an erosion, which removes tiny segments. The closing operation is an erosion followed
by a dilation. This removes tiny gaps in otherwise filled regions. They have been used previously
for biomedical image segmentation.

- Opening/closing morphological operations to process the predicted segmentation mask.
- Simple denoising strategy to refine the final predictions based on adjacent pixels.
- Implemented but with no/slight improvement: CRF as final post-processing step.

## Figures

#### Image pre-processing using traditional segmentation approaches.
![Image pre-processing](https://github.com/VKonstantakos/EY-Data-Challenge-2021/blob/main/media/preprocessing.png)

#### Architecture of various segmentation models.
![Segmentation models](https://github.com/VKonstantakos/EY-Data-Challenge-2021/blob/main/media/model.png)

## Results

With F<sub>1</sub> score  as the evaluation metric, the models performed as follows:

| Base Models    | Training   | Validation |
| -------------- | ------ | -------- |
| FPN - Jaccard | **70.4%** | **63.4%**  |
| FPN - Tversky | 51.0% |50.2% |
| FPN - BCE-Dice | 64.7% | 60.7%  |
| U-net Light - Jaccard | 59.8% | 55.6%  |

FPN: Feature Pyramid Network, BCE: Binary Cross-Entropy, U-net Light: Custom U-net architecture.

### Submission Result

| Final submission   | Public LB   | Private LB | National Ranking | European Ranking |
| -------------- | -------- | -------- | -------- | -------- |
| LoG &rarr; FPN &rarr; Denoise| 71.09% | 71.08%  | 1st | Top 10|



## Support

For questions, email vkonstantakos@iit.demokritos.gr

