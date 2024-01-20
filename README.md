# Normal Flow Prediction
This project consists of a deep learning model able to predict the normal flow between two consecutive frames, being the normal flow the projection of the optical flow on the gradient directions. The dataset used to train the model has been [TartanAir dataset](https://theairlab.org/tartanair-dataset/) and the deep learning model is an encoder-decoder with residual blocks based on [EVPropNet](https://prg.cs.umd.edu/EVPropNet).

## Normal Flow
The normal flow is the projection of the optical flow on the gradient directions of the image and serves as a representation of image motion. To compute it, the [brightness constancy constraint](https://www.cs.toronto.edu/~fleet/research/Papers/flowChapter05.pdf) has applied. The brightness constancy constraint is one of the fundamental assumptions in optical flow computation and computer vision. It is based on the idea that, the intensity, or a function of the intensity, at a pixel remains constant over two consecutive frames. The mathematical expression of this constraint is:
$I(x,y,z)=I(x+\u\delta t)$
