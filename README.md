# Normal Flow Prediction
This project consists of a deep learning model able to predict the normal flow between two consecutive frames, being the normal flow the projection of the optical flow on the gradient directions. The dataset used to train the model has been [TartanAir dataset](https://theairlab.org/tartanair-dataset/) and the deep learning model is an encoder-decoder with residual blocks based on [EVPropNet](https://prg.cs.umd.edu/EVPropNet).

## Normal Flow
The normal flow is the projection of the optical flow on the gradient directions of the image and serves as a representation of image motion. To compute it, the [brightness constancy constraint](https://www.cs.toronto.edu/~fleet/research/Papers/flowChapter05.pdf) has to be applied. The brightness constancy constraint is one of the fundamental assumptions in optical flow computation and computer vision. It is based on the idea that, the intensity, or a function of the intensity, at a pixel remains constant over two consecutive frames. The mathematical expression of this constraint is:
<p align="center">
$I(x,y,z)=I(x + u \delta t, y + v \delta t, t + \delta t)$
</p>

where $u$ and $v$ represent the optical flow of the pixel $(x, y)$ (i.e., the motion of the image pixel from time $t$ to $t+1$). Thus, the equation can be rewritten as:
<p align="center">
$I(x,y,z)=I(x + \delta x, y + \delta y, t + \delta t)$
</p>

Approximating the right part of the previous equation with a first order Taylor expansion we obtain:
<p align="center">
$I(x,y,z)=I(x,y,z) + \frac{\partial I}{\partial x}\delta x + \frac{\partial I}{\partial y}\delta y + \frac{\partial I}{\partial t}\delta t$
</p>

And subtracting $I(x,y,t)$ from both sides of the equation, and then dividing it by $\delta t$:
<p align="center">
$0 = \frac{\partial I}{\partial x} \frac{\delta x}{\delta t} + \frac{\partial I}{\partial y} \frac{\delta y}{\delta t} + \frac{\partial I}{\partial t} \frac{\delta t}{\delta t} = I_x \frac{\partial x}{\partial t} + I_y \frac{\partial y}{\partial t} + I_t$
</p>

Finally, keeping in mind that the optical flow $(u,v)$ represents the motion of the image pixel from time $t$ to $t+1$, this last equation can be rewritten as:
<p align="center">
$0 = I_x u + I_y v + I_t$
</p>

This equation represents the constraint line. For any point $(x,y)$ in the image, its optical flow $(u,v)$ lies on this line. In the following image, it can be seen an example of this line alongside an optical flow vector. As shown in the image, the optical flow vector (blue arrow) can be decomposed into two components: the normal flow (depicted by the red arrow) and the parallel flow (indicated by the green arrow).
<p align="center">
<img src="https://github.com/FandosA/Normal_Flow_Prediction/assets/71872419/5142de5e-31dc-4567-85c4-926a8c145837" width="350" height="350">
</p>

Therefore, to compute the normal flow vector, it is necessary to calculate the unit vector perpendicular to the constraint line and its magnitude, which corresponds to the distance from the origin to the constraint line. The mathematical expressions of this components are:
<p align="center">
$\hat{u}_n = \frac{(I_x, I_y)}{\sqrt{I_x^2 + I_y^2}}$
</p>
<p align="center">
$|\hat{u}_n| = \frac{|I_t|}{\sqrt{I_x^2 + I_y^2}}$
</p>

Obtaining $I_t$ from the equation, $-I_t = I_xu + I_yv$, the final normal flow vector can be calculated by combining the unit vector of the normal flow, its magnitude, and the value of $|I_t|$:
<p align="center">
$u_n = |\hat{u}_n| \hat{u}_n = \frac{|I_t|}{\sqrt{I_x^2 + I_y^2}} \cdot \frac{(I_x,I_y)}{\sqrt{I_x^2 + I_y^2}} = \frac{I_xu + I_yv}{I_x^2 + I_y^2} (I_x,I_y)$
</p>

## Autoencoder
The deep learning model chosen to predict the normal flow between two consecutive frames has been an autoencoder. This autoencoder is based on [EVPropNet](https://prg.cs.umd.edu/EVPropNet), which in turn is based on [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). The encoder contains residual blocks with convolutional layers and the decoder contains residual blocks with transpose convolutional layers. The gradients are backpropagated using a mean squared loss computed between groundtruth and predicted normal flow:
<p align="center">
$argmin$ $||n - \hat{n} ||_2^2$
</p>

The model takes as input two concatenated frames and outputs a matrix of two channels, the components of the normal flow of each pixel. That is, the dimensions of the input and output tensors are $(h,w,6)$ and $(h,w,2)$, respectively.

## Run the implementation
As mention before, The dataset used to train the model has been [TartanAir dataset](https://theairlab.org/tartanair-dataset/). This dataset provides many image sequences of different scenarios created in Unreal Engine. At the same time they provide depth maps, optical flow, camera positions and orientations in each image and more. you need to visit their website, download the scenarios you want, and organize the images and their optical flow data the same way they are here in the ```dataset/train``` folder in the repository. When the data is correctly organized, run the file
```
python dataset.py
```
This will create a _json_ file like the one here in the repository with the paths to all images and optical flow data. Then run
```
python train.py
```
and the model will start training. A folder like the one here called ``autoencoder`` will be created. The training checkpoints, as well as the loss values, will be saved here. At the end of the training, an image showing the loss curves will also be saved. You can check the folder in this repository to see what it looks like and the loss curves I have obtained.

To test the model, organise the dataset in the same way as before but using the ```dataset/test``` folder instead, and run the test file
```
python test.py
```
You only have to enter the name of the checkpoint you want to use. In my case, I run
```
python test.py --checkpoint=checkpoint_395_best.pth
```
because that's the name of the checkpoint where the loss value was the lowest in my training.
