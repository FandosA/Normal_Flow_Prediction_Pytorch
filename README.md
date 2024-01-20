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

