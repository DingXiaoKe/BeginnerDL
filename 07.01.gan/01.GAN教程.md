对抗生成网络
===
# 1.对抗生成网络的分类
对抗生成网络(GAN：Generative Adversarial Nets)可以分为DCGAN、CGAN、StackGAN、infoGANs、Wasserstein GAN、DiscoGAN、BEGAN等等

# 2.对抗生成网络介绍
对抗生成网络简单来说就是以下三点：
- 构建两个网络，一个G网络，一个D网络。两个网络的网络结构随意就好
- 训练方式。G网络的loss是$log(1-D(G(z)))$，而D网络的loss是$-(log(D(x)) + log(1-D(G(z))))$，注意这里并不是Cross Entropy
- 数据输入。G网络的输入是Noise，而D网络的输入则是混合G的输出数据及样本数据

G网络的训练是希望$D(G(z))$趋近于1，这样G的loss就会最小。而D网络是一个二分类，目标是分清楚真实数据和假数据，也就是希望真实数据的D输出趋近于1，而生成数据的输出即D(G(z))趋近于0。

# 3.训练方法
- G和D是同步训练的，但是两者的训练次数不一样，G训练一次，D训练k次。这主要还是因为初代的GAN训练不稳定。
- 注意D的训练是同时输入生成的数据和样本数据计算loss，而不是cross entropy分开计算。实际上为什么GAN不用cross entropy是因为，使用cross entropy会使$D(G(z))$变为0，导致没有梯度，无法更新G，而GAN这里的做法$D(G(z))$最终是收敛到0.5。
- 在实际训练中，文章中G网络使用了RELU和sigmoid，而D网络使用了Maxout和dropout。并且文章中作者实际使用$-log(D(G(z))$来代替$log(1-D(G(z))$，从而在训练的开始使可以加大梯度信息，但是改变后的loss将使整个GAN不是一个完美的零和博弈。

G、D同步训练，G训练一次，D训练k次。D训练同量输入生成数据和样本数据计算loss(不是cross entropy分开计算)。cross entropy使D(G(z))为0,导致没有梯度，无法更新G。GAN $D(G(z))$最终收敛到0.5。G网络用RELU、sigmoid，D网络用Maxout和dropout。$-log(D(G(z)))$代替$log(1-D(G(z)))$，训练开始加大梯度信息，整个GAN不是完美零和博弈。
