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

And subtracting $I(x,y,t)$ from both sides of the equation, and dividing it by $\delta t$:

<p align="center">
$0 = \frac{\partial I}{\partial x}\delta x + \frac{\partial I}{\partial y}\delta y + \frac{\partial I}{\partial t}\delta t$
</p>

