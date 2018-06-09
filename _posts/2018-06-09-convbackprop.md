---
title: "Understanding Backpropagation for ConvNets"
date: 2018-06-09
tags: [deep learning, computer vision, backpropagation, convolutional neural networks, cnn, convnet, convolution, neural network,
      gradient descent]
header:
  image: "/images/convbackprop/header.png"
excerpt: "Deep Learning, Convolutional Neural Network (CNN), Backpropagation"
mathjax: "true"
---

Convolutional Neural Networks or ConvNets are everywhere these days thanks to its extensive application in a wide range of tasks starting from toy examples such as dogs vs cat classifications to much more intricate autonomous cars and likes. In fact, it is also pretty easy to learn the basics of these networks and understand how they outperform fully-connected networks in solving advanced problems. A series of great tutorials are available online which include masterpieces like [Andrew Ng’s deeplearning.ai in Coursera](https://www.coursera.org/specializations/deep-learning), [Jeremy Howard’s Fast.ai](http://course.fast.ai/), [Udacity’s Deep Learning nanodegree](https://in.udacity.com/course/deep-learning-nanodegree--nd101) and of course the famous [cs231n by Stanford](http://cs231n.stanford.edu/). In case you are just starting out your journey in the deep learning space, I would highly recommend you check out the above courses.

What I found to be not so trivial however, perhaps even after you are off the ground with a few deep learning models in your pocket, is understanding how backpropagation work for these networks. That being said, I assume that you are fairly familiar with the backpropagation algorithm for shallow neural networks. Again, if you are not so comfortable with the weight updates through gradients in different layers, please check out the above courses. [This article by Michael Nielsen](http://neuralnetworksanddeeplearning.com/) goes pretty deep into the underlying math of neural networks in case you are interested.

This post, however, is not intended to derive the mathematical equations for the backpropagation for convolutional networks. Rather, it is aimed at providing you with a conceptual clarity about how it works. By the end of this article, you should be in a comfortable position to do the math by yourself with your favourite notational nuance. In the following section, we shall see how the forward and backward pass look like for a convolutional layer and how it differs from a fully-connected layer. Just one caveat before we dive in, this is my first try to write a blog post, so please bear with me in case something is not well conveyed or overemphasised.


Instead of trying to master the backpropagation for a particular type of network — what I found more useful is to have a clear idea aboutf what a backward pass would achieve given a forward pass. It really helps to think about the network as a computation graph where we feed in certain inputs which propagate forward through the layers to generate an output, then we use that output to calculate a cost function (you can think of it as a feedback from the labels for the current pass) and then calculate the gradients for that cost function which in turn propagate back through the network adjusting the weights (and biases) up till the first layer. As such, we can calculate the activation of a particular layer given some weights and biases and the activation of the previous layer in our network. What is more interesting is the fact that we can also calculate the gradients of the weights as well as the gradient for the previous layer activation given the gradient of the current layer. Let’s look at the figure below:

<img src="{{ site.url }}{{ site.baseurl }}/images/convbackprop/backprop_cs231n.png" alt="forward and backward pass of a hidden layer unit">

Here's some math:

$${A^l} = {W^l}{A^{l - 1}} + {b^l}$$
