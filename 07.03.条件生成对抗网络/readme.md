条件生成对抗网络-cGAN(Conditional Generative Adversarial Nets)
===
# 1.MNIST数据集
在生成器和判别器中都加入label的信息，pytorch框架训练20个epoch之后的结果如下图<br/>
![images](results/MNIST_cDCGAN_20.png)<br/>
![images](results/pytorch_mnist.gif)<br/>
Loss的变化如下图所示:<br/>
![images](results/MNIST_cDCGAN_train_hist.png)
生成模型pth文件为pytorch_mnist_generator_cdan.pth,在[百度云盘](https://pan.baidu.com/s/1JWLMbibaH1yGZKcIvyT4hQ#list/path=%2F%E6%A8%A1%E5%9E%8B)

Keras版本的结果如下<br/>
![images](results/keras_mnist_10000.png)<br/>
![images](results/keras_mnist.gif)<br/>
生成模型h5文件为keras_mnist_generator_cdan.h5,在[百度云盘](https://pan.baidu.com/s/1JWLMbibaH1yGZKcIvyT4hQ#list/path=%2F%E6%A8%A1%E5%9E%8B)

# 2.Cifar10数据集
![images](results/pytorch_cifar10_cdan.png)<br/>
![images](results/pytorch_cifar10_cdan.gif)<br/>
生成模型pth文件为pytorch_cifar10_cdan_G_19.pth,在[百度云盘](https://pan.baidu.com/s/1JWLMbibaH1yGZKcIvyT4hQ#list/path=%2F%E6%A8%A1%E5%9E%8B)
