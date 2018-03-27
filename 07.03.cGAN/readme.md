条件生成对抗网络-cGAN(Conditional Generative Adversarial Nets)
===
# 1.MNIST数据集
在生成器和判别器中都加入label的信息，pytorch框架训练20个epoch之后的结果如下图<br/>
![images](results/MNIST_cDCGAN_20.png)
![images](results/pytorch_mnist.gif)<br/>
Loss的变化如下图所示:
![images](results/MNIST_cDCGAN_train_hist.png)

Keras版本的结果如下<br/>
![images](results/keras_mnist_9800.png)<br/>
![images](results/keras_mnist.gif)

# 2.
