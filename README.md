# nanoGPT

Create a generative pre-trained transformer neural network following the video by Andrej Karpathy from January 2023. [ChatGPT](https://chatgpt.com/) had just been released a few weeks earlier on November 30, 2022 by OpenAI. Andrej is one of the co-founders of OpenAI. The first model, [GPT-1](https://en.wikipedia.org/wiki/GPT-1) was released in June 2018.

## Source

Video: [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) from January 2023 for infinite Shakespeare 1:56:19.

#### Links

- [Google colab for the video](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- [GitHub repo for the video](https://github.com/karpathy/ng-video-lecture) as video lecture companion
- [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) updated project __nanoGPT__ starting December 2022, ChatGPT was just released
- [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT) earlier idea with just 300 lines of Python code starting August 2020 to June 2022

#### Papers

- Attention Is All You Need (2017) [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Language Models are Few-Shot Learners (2020) [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

## History

This is just for my own sanity. A brief overview of the development of machine learning ML, image classification, object detection, self-driving car expectations, generative pre-trained transformer GPT, AI, AGI, general and generative AI. The first GPT was published by OpenAI in 2018 following the 2017 paper *Attention Is All You Need*.

## Manual labeling images for Machine Learning - 2007 Image Net

As [Fei Fei Li](https://en.wikipedia.org/wiki/Fei-Fei_Li) explains in her [TET talk in 2015](https://www.youtube.com/watch?v=40riCqvRoMs) the group at Stanford started in 2007 to classify around 1,000,000,000 pictures and used [Amazon Mechanical Turk](https://en.wikipedia.org/wiki/Amazon_Mechanical_Turk) with 48,940 workers in 167 countries to create the image database [ImageNet](https://en.wikipedia.org/wiki/ImageNet). By 2009 the database [https://www.image-net.org/](https://www.image-net.org/) had 14,197,122 labeled images in 21,841 categories. And funding back then was a problem!

## ILSVRC competition - 2010

The initial approach was to look for features manually in the pictures (hard-coded) to classify them correctly. The top-5 error rate in 2010 was 28.2% with this approach. A deep neural network for deep learning would make big progress 2 years later!

![2015 ILSVRC](https://github.com/kreier/nano-gpt/blob/main/docs/2015-ILSVRC.png?raw=true)

## Progress in Image Classification - 2012 AlexNet with a convolutional neural network - CNN

To win the ImageNet 2012 challenge [AlexNet](https://en.wikipedia.org/wiki/AlexNet) used __GPUs__ for its concolutional neural network CNN. Its running on CUDA. This competition ILSVRC started in 2010 and ran until 2017. The creator of ImageNet Fei Fei Li gave an [inspiring TED talk in 2015](https://www.youtube.com/watch?v=40riCqvRoMs) including the AlexNet solution. Self driving cars are mentioned at minute 1:30. And how much a three-year old can outperform a computer. And Elon Musk is talking since 2014 that autonomous driving is just a year away.

![AlexNet 2012](https://github.com/kreier/nano-gpt/raw/main/docs/2012_AlexNet.png)

## Deep Learning for Computer Vision - 2016 Andrej Karpathy

A [great talk about Deep Learning](https://www.youtube.com/watch?v=u6aEYuemt0M) by Andrej Karpathy from September 25, 2016 explains in detail and clarity the different layers in AlexNet and further develpments.

| layer     | size      | architecture                        |    memory | parameter |
|-----------|-----------|-------------------------------------|----------:|----------:|
| INPUT     | 227x227x3 |                                     |   154,587 |         0 |
| CONV1     | 55x55x96  | 96 11x11 filters at stride 4, pad 0 |   290,400 |    11,616 |
| MAX POOL1 | 27x27x96  | 3x3 filters at stride 2             |    69,984 |         0 |
| NORM1     | 27x27x96  | Normalization layer                 |    69,984 |         0 |
| CONV2     | 27x27x256 | 256 5x5 filters at stride 1, pad 2  |   186,624 |     6,400 |
| MAX POOL2 | 13x13x256 | 3x3 filters at stride 2             |    43,264 |         0 |
| NORM 2    | 13x13x256 | Normalization layer                 |    43,264 |         0 |
| CONV3     | 13x13x384 | 384 3x3 filters at stride 1, pad 1  |    64,896 |     3,456 |
| CONV4     | 13x13x384 | 384 3x3 filters at stride 1, pad 1  |    64,896 |     3,456 |
| CONV5     | 13x13x256 | 256 3x3 filters at stride 1, pad 1  |    43,264 |     2,304 |
| MAX POOL3 | 6x6x256   | 3x3 filters at stride 2             |     9,216 |         0 |
| FC6       | 4096      |                                     |     4,096 |         0 |
| FC7       | 4096      |                                     |     4,096 |         0 |
| FC8       | 1000      | 100 neurons (class scores)          |     1,000 |         0 |
|           |           |                                     | 1,049,571 |    27,232 |

Source material from Stanford:

- [https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) ConvNetJS CIFAR-10 demo needs a little time to download in the background and start
- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/) - Deep Learning in your browser
- [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) at Stanford

## Object detection - 2017 YOLO

The [TED Talk from August 2017](https://www.youtube.com/watch?v=Cgxsv1riJhI) by Joseph Redmon from Washington University inspired ideas and possibilities. In his talk he talked about the application for self driving cars. It certainly makes it imaginable, and it was just running on his laptop! 

## The transformer model of machine learning and neuronal networks - June 2017

- Attention Is All You Need (2017) [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

<img src="https://raw.githubusercontent.com/ssis-robotics/.github/main/profile/transparent.png" width="20%"> <img src="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png" width="60%">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Full_GPT_architecture.png/399px-Full_GPT_architecture.png" width="30%" align="right">

## First GPT released by OpenAI - June 2018

OpenAI released its first Generative Pre-trained Transformer 1 (GPT-1) following the transformer architecture published by Google in 2017. [https://openai.com/index/language-unsupervised/](https://openai.com/index/language-unsupervised/)

## Application in diploma thesis in Saigon - 2018

A student at [HCMUTE](https://en.wikipedia.org/wiki/Ho_Chi_Minh_City_University_of_Technology_and_Education) used this [YOLO](https://www.youtube.com/watch?v=MPU2HistivI) (You Only Look Once) software with an NVidia graphics card in his master thesis 2018 for an part in developing an autonomous car. But the [GTX 980](https://en.wikipedia.org/wiki/GeForce_900_series) with 2048 CUDA cores need 165 Watt power to work. The power requirement is a challenge for a mobile application.

## Tensorflow.lite and posenet on new RX 470 GPU - March 2019

The crypto-boom was over and graphics cards available again. So I started with an RX 470 in March 2019. In October 2018 I had already successfully installed Darknet on Ubuntu. The new repository [https://github.com/kreier/ml](https://github.com/kreier/ml)

## Mobile affordable platform - 2019 Jetson Nano

Nvidia had the Jetson TK1 platform already created in 2014, but it was power hungry and expensive. But for just 100 Dollar the Jetson Nano was announced in 2019 to be affordable for student projects with just using 5-10 Watt. We applied for a project at hackster.io. But ultimately we ordered a 4GB development platfrom at the end of 2019 for student projects.

The drive base from TAE was delayed early 2020 because of the starting COVID-19 pandemic and shipments from China. But eventually we got the drive base. Yet the project stalled from 2020-2024.

![Jetson and car](https://github.com/kreier/jetson-car/blob/main/pic/2019_jetson_car.jpg?raw=true)

## OpenAI releases ChatGPT and the world takes note - November 2022

The underlying model GPT-3.5 was already released in March 2022. But in order to make it an assistant you can talk to some __finetuning__ had to be done. See blog post [https://openai.com/index/chatgpt/](https://openai.com/index/chatgpt/)

![Finetuning ChatGPT](https://github.com/kreier/nano-gpt/blob/main/docs/finetuning.jpg?raw=true)

And ever since the interface was available on the web with ChatGPT since November 2022 the world took note. It ran in news outlets and personal conversations worldwide and within two years NVIDIA became the most valuable company in the world.

## Challenges of self driving - 2024

It seemed like the combination of object detection and classification was the main problem to get self driving cars possible, together with latency. As it turns out, there is much more to do. Additinal sensors like LIDAR really help, even Tesla finally gave in. And here are some example videos to show whats possible and what is still challenging:

- [Why self-driving cars have stalled | It's Complicated](https://www.youtube.com/watch?v=4sCK-a33Nkk) by The Guardian September 2022
- [I RACED My Self-Driving RC Car](https://www.youtube.com/watch?v=zuyOdaQ2xuw) by Steven Gong in July 2023, taking part in the [F1 tenth](https://f1tenth.org/) race 2023
- [Sorry. Your Car Will Never Drive You Around.](https://www.youtube.com/watch?v=2DOd4RLNeT4) - from January 2024, self driving cars are 8times __less__ save than humans, see data at minute 15 and 11.3 deaths per 100 million miles for Tesla FSD versus 1.35 deaths for humans
- [Video First AI powered Race - A2RL Abu Dhabi](https://www.youtube.com/watch?v=feTxamTHQAA) - dissaster 2024-04-28 [https://a2rl.io/](https://a2rl.io/)

## Compute power

- __GPT-1__ needed `8.6e19` FLOP
- __GPT-2__ needed `1.5e21` FLOP
- __GPT-3__ needed `3.1e23` FLOP
- __GPT-3.5__ undisclosed
- __GPT-4__ needed `2.1e25` FLOP estimated
- __3070 Ti__ running 24h: 2.3e13 FLOPS x 86400 = `1.9e18` - not even enough for GPT-1 (according to [test with OpenCL](https://github.com/kreier/benchmark/tree/main/gpu/opencl)), that would require 45 days!

__First run__ at [1:21:28](https://youtu.be/kCc8FmEb1nY?si=veYi24diWjjoPKMB&t=4888) with `train loss 2.39` and `val loss 2.41`. No time mentioned, but here are some parameters:

``` py
batch_size = 32
block_size = 8
learning_rate = 1e-3
n_embd = 32
```

__Multi-Head Attention__ at 1:24:24 improves to `train loss 2.27` and `val loss 2.28`.

__Improved training:__ The adjusted parameters (residual connections, skip and layernorm) are in the lines  `144` introduce n_layer and n_head, `145` pull out LayerNorm,  `113` include dropout, __1:39:34__ `hyperparameters` batch_size = 64, block_size = 256, learning_rate = 3e-4, n_embd = 384, n_head = 6 so each head has 384/6 = 64 dimensions, n_layer = 6 to have a `val loss 1.48`. In overview:

``` py
batch_size = 64       # from 32
block_size = 256      # from 8
learning_rate = 3e-4  # from 1e-3
n_embd = 384          # from 32
```

According to [1:40:46](https://youtu.be/kCc8FmEb1nY?si=3yNr-iwpgYe5imeX&t=6046) this leads to 15 minutes on a [A100 GPU](https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623). Let's assume its a 40 GB version with 156 TFLOPS on TF32 this needed 1.4e17 FLOP. If not constrained by having only 8GB of GDDR6X RAM then this should finish on my 3070 Ti after 6100 seconds or 1h40. Or with a price of $2.00 per hour for an A100 I could just rent it for $0.50 to do the calculation.
