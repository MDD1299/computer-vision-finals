# Generative AI: GANs Implementation on Flask Framework

## As part of the Final Activity for our subject Computer Vision

### Introduction
Generative Adversarial Networks (GANs) represent a cutting-edge approach in generative modeling within deep learning. These networks, often based on convolutional neural networks, aim to autonomously discern patterns in input data, allowing the generation of new examples closely resembling the original dataset. GANs involve two key components: the generator, creating novel examples, and the discriminator, distinguishing between genuine and generated instances. Through adversarial training, these models engage in a competitive interplay, resulting in the generator producing realistic samples, challenging the discriminator approximately half of the time (GeeksforGeeks, 2023).

### Review of Related Literature
1. **Unsupervised Representation Learning with DCGANs**
2. **Facial Parts Responses to Face Detection**
3. **Joint Face Detection and Alignment using MTCNN**

### Course Topic
This course focuses on developing and deploying a GAN model for generating realistic faces using the CelebA Face Dataset, with a final implementation in Flask. The course employs Python, TensorFlow, Flask, and HTML/CSS, providing a practical skill set in deep learning and web development. The course covers two key models: the Generator and the Discriminator.

- **Generator Model:** Neural network creating synthetic data, transforming random noise into realistic samples.
- **Discriminator Model:** Evaluates and classifies input data as real or generated. Trains simultaneously with the Generator in an adversarial loop.

### Data Used Information
The CelebFaces Attributes Dataset (CelebA) comprises over 200,000 images of celebrities, annotated with 40 attribute labels. The dataset provides rich information, including landmark locations and binary attributes (Tensorflow, n.d.).

For access to the dataset, visit: [CelebA dataset link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### Problem Statement
The primary objective is to train a GAN model proficient at generating images. GANs, consisting of a generator and a discriminator, work in tandem, with the generator producing realistic data and the discriminator distinguishing between authentic and generated content. The iterative and adversarial training process refines the generator's ability to create authentic images while enhancing the discriminator's discrimination capabilities.

### Method
#### Algorithm:
The GAN-based approach uses convolutional neural networks (CNNs) in both the generator and discriminator. Training involves optimizing these networks adversarially to generate realistic images from random noise.

#### Solution:
The solution implements a GANs model for image generation. The adversarial training loop allows the generator to refine its capabilities, creating increasingly realistic images, while the discriminator sharpens its ability to differentiate between genuine and synthetic visuals.

### Evaluation
The model's performance is comprehensively evaluated using accuracy metrics on real and fake images. Achieving an average accuracy above 85% indicates the GAN has successfully learned to generate synthetic images closely resembling real ones.

### Results & Discussion
**Part 1: GANs Model Creation**

**Part 2: Web Application Creation using Flask Framework**

### References
- GeeksforGeeks. (2023, November 23). [Generative Adversarial Network GAN](https://www.geeksforgeeks.org/generative-adversarial-network-gan/).
- Radford, A., Metz, L., & Chintala, S. (2015). [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/abs/1511.06434).
- Yang, S., Luo, P., Loy, C. C., & Tang, X. (2015). [From facial parts responses to face detection: A deep learning approach](https://openaccess.thecvf.com/content_iccv_2015/html/Yang_From_Facial_Parts_ICCV_2015_paper.html).
- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). [Joint face detection and alignment using multitask cascaded convolutional networks](https://ieeexplore.ieee.org/document/7553523).
- Tensorflow. (n.d.). [CelebA](https://www.tensorflow.org/datasets/catalog/celeb_a).
