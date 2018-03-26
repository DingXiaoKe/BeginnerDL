DCGAN(Deep Convolutional Generative Adversarial Networks)
===
在GAN的内部使用了卷积神经网络，并且实现了有效训练。它进行了以下几个有效的改进：
- 去掉了G网络和D网络中的pooling layer
- 在G网络和D网络中都使用Batch Normalization
- 去掉全连接的隐藏层
- 在G网络中除最后一层使用RELU，最后一层使用Tanh
- 在D网络中每一层使用LeakyRELU

# 1.具体网络结构
- G网络：100 z->fc layer->reshape ->deconv+batchNorm+RELU(4) ->tanh 64x64
- D网络版本1：conv+batchNorm+leakyRELU (4) ->reshape -> fc layer 1-> sigmoid
- D网络版本2：conv+batchNorm+leakyRELU (4) ->reshape -> fc layer 2-> softmax

G网络使用4层反卷积，而D网络使用了4层卷积。基本上G网络和D网络的结构正好是反过来的。那么D网络最终的输出有两
种做法，一种就是使用sigmoid输出一个0到1之间的单值作为概率，另一种则使用softmax输出两个值，一个是真的概率，
一个是假的概率。两种方法本质上是一样的