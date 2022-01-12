# im2normals
Using Deep Learning to estimate surface normals using RGB images

This work was done as part of the Computation Vision course at Princeton University, under the supervision of professor Jia Deng. An initial template script was provided by the course supervisors to load/read the data.

Development notes
* Split the provided 20K data tuples of color image, mask and normal images  into training and validation set by using first 19K images for training and last 1K images for validation.
* Replaced the code in build_model() with Fully Convolutional DenseNet by modifying the publicly available code [here](https://github.com/HasnainRaz/FC-DenseNet-TensorFlow/blob/master/model.py).
* Network architecture
  - Model was created as DenseTiramisu(12, [4,4,4,4,4]) i.e. FCDenseNet-56, which has 4 levels of downsampling and upsampling. Reference included in Page 4. I could not try larger networks (67, 103) since training was very slow.
  - Removed batch normalization from all the layers. This was essential otherwise the network was not training well and generated normal images were very noisy.
  - Dropout rate of 0.2 was used (as set in the original code)
  - Used Adam solver with initial learning rate of 10^{-4}.
* Ran experiments with batch size 20 for 150K iterations, evaluating on validation set at every 500 iterations. The best model was selected based on the performance on the validation set.
