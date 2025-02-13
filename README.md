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

## Hardware requirements

While the project can be compiled even on a CPU for best results you would use a GPU. But not any GPU, it needs to be from Nvidia to use the CUDA capabilities from `torch`. The software stack for AMD is still a WIP. And I have several Nvidia GPUs. To test your own setup under Windows with WSL and Ubuntu 22.04 LTS (don't use 24.04 LTS since it has python 3.12 which is a little to new) with the following few lines:

``` sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
git clone https://github.com/karpathy/nanogpt
cd nanogpt
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

On my slightly older GTX 960 I get the warning after the training call:

``` sh
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Found NVIDIA GeForce GTX 960 which is too old to be supported by the triton GPU compiler, which is used
as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 5.2
```

Let's see what I have and what CUDA capabilities these support:

| GPU name     | CUDA cores | Compute Capability |      at     | architecture | RAM GB |
|--------------|-----------:|:------------------:|:-----------:|--------------|-------:|
| Quadro FX580 |         32 |         1.1        | hp Z600     | [Tesla](https://en.wikipedia.org/wiki/Tesla_(microarchitecture)) (2006) |    0.5 |
| GTX 650      |        384 |         3.0        | hp Z600     | [Kepler](https://en.wikipedia.org/wiki/Kepler_(microarchitecture)) (2012) |     1 |
| Jetson Nano  |        128 |         5.3        |             | [Maxwell](https://en.wikipedia.org/wiki/Maxwell_(microarchitecture)) (2014) |    4 |
| GT750M       |        384 |         3.0        | MBPr15 2014 | Kepler (2012) |   0.5 |
| M1000M       |        512 |         5.0        | Zbook 15 G3 | Kepler (2012) |     1 |
| GTX960       |       1024 |         5.2        | E5-2696 v3  | Maxwell (2014) |    2 |
| T4           |       2560 |         7.5        | Google Colab | Turing (2018) |   16 |
| RTX3070 Ti   |       6144 |         8.6        | i3 10100    | Ampere (2020)  |    8 |

Only one of 7 is supported by the Triton GPU compiler. How about a newer GPU?

| GeForce series | CUDA | Architecture | Process | Year |
|----------------|------|--------------|:-------:|------|
| 900            | 5.2  | [Maxwell](https://en.wikipedia.org/wiki/Maxwell_(microarchitecture))      | 28HP    | 2014 |
| 10             | 6.1  | [Pascal](https://en.wikipedia.org/wiki/Pascal_(microarchitecture))        | 16FF    | 2016 |
| 16             | 7.5  | [Turing](https://en.wikipedia.org/wiki/Turing_(microarchitecture))        | 12FFN   | 2018 |
| 20             | 7.5  | Turing       | 12FFN   | 2018 |
| 30             | 8.6  | [Ampere](https://en.wikipedia.org/wiki/Ampere_(microarchitecture))        | 8LPP    | 2020 |
| 40             | 8.9  | [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) | 4N | 2022 |
| 50             | 10.0 | [Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))  | 4NP     | 2024 |

Looks like at least series 16 or 20, but probably 30 to be sure when future compilers increase to 8.0.

## History

This is just for my own sanity. A brief overview of the development of machine learning ML, image classification, object detection, self-driving car expectations, generative pre-trained transformer GPT, AI, AGI, general and generative AI. The first GPT was published by OpenAI in 2018 following the 2017 paper *Attention Is All You Need*.

## Manual labeling images for Machine Learning - 2007 Image Net

[Fei Fei Li](https://en.wikipedia.org/wiki/Fei-Fei_Li) explains in her [TET talk in 2015](https://www.youtube.com/watch?v=40riCqvRoMs) that her group at Stanford started in 2007 to classify around 1,000,000,000 pictures and used [Amazon Mechanical Turk](https://en.wikipedia.org/wiki/Amazon_Mechanical_Turk) with 48,940 workers in 167 countries to create the image database [ImageNet](https://en.wikipedia.org/wiki/ImageNet). By 2009 the database [https://www.image-net.org/](https://www.image-net.org/) had 14,197,122 labeled images in 21,841 categories. And funding back then was a problem!

## ILSVRC competition - 2010

The initial approach was to look for features manually in the pictures (hard-coded) to classify them correctly. The top-5 error rate in 2010 was 28.2% with this approach. A deep neural network for deep learning would make big progress 2 years later!

![2015 ILSVRC](https://github.com/kreier/nano-gpt/blob/main/docs/2015-ILSVRC.png?raw=true)

## Progress in Image Classification - 2012 AlexNet with a convolutional neural network - CNN

To win the ImageNet 2012 challenge [AlexNet](https://en.wikipedia.org/wiki/AlexNet) used __GPUs__ for its concolutional neural network CNN. Its running on CUDA. This competition ILSVRC started in 2010 and ran until 2017. The creator of ImageNet Fei Fei Li gave an [inspiring TED talk in 2015](https://www.youtube.com/watch?v=40riCqvRoMs) including the AlexNet solution. Self driving cars are mentioned at minute 1:30. And how much a three-year old can outperform a computer. And Elon Musk is talking since 2014 that autonomous driving is just a year away.

![AlexNet 2012](https://github.com/kreier/nano-gpt/raw/main/docs/2012_AlexNet.png)

## Deep Learning for Computer Vision - 2016 Andrej Karpathy

A [great talk about Deep Learning](https://www.youtube.com/watch?v=u6aEYuemt0M) by Andrej Karpathy from September 25, 2016 explains in detail and clarity the different layers in AlexNet and further develpments. Here is an overview of the layers for AlexNet, and a [benchmark tests](https://github.com/MrYxJ/calculate-flops.pytorch) the FLOPS for each layer with your current GPU is shown below: 

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

Benchmark run [calflops](https://pypi.org/project/calflops/) (and with this model far from the 22 TFLOPS possible):

``` sh
(venv) mk@i3:~/test$ python pytorch-calflops.py

------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  61.1 M
fwd MACs:                                                               714.188 MMACs
fwd FLOPs:                                                              1.4297 GFLOPS
fwd+bwd MACs:                                                           2.1426 GMACs
fwd+bwd FLOPs:                                                          4.2892 GFLOPS
---------------------------------------------------------------------------------------------------
Alexnet FLOPs:1.4297 GFLOPS   MACs:714.188 MMACs   Params:61.1008 M
```

Source material from Stanford:

- [https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) ConvNetJS CIFAR-10 demo needs a little time to download in the background and start
- [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/) - Deep Learning in your browser
- [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) at Stanford

## Object detection - 2017 YOLO

The [TED Talk from August 2017](https://www.youtube.com/watch?v=Cgxsv1riJhI) by Joseph Redmon from Washington University inspired ideas and possibilities. In his talk he talked about the application for self driving cars. It certainly makes it imaginable, and it was just running on his laptop! 

<img src="https://raw.githubusercontent.com/kreier/nano-gpt/main/docs/yolo1.jpg" width="28%"> <img src="https://raw.githubusercontent.com/kreier/nano-gpt/main/docs/yolo2.jpg" width="70%">

<img src="https://raw.githubusercontent.com/kreier/jetson-car/main/pic/white10x1.png" width="25%"> <img src="https://raw.githubusercontent.com/kreier/nano-gpt/main/docs/yolo-timeline.png" width="50%">

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

The drive base from TAE was delayed early 2020 because of the starting COVID-19 pandemic and shipments from China. But eventually we got the drive base. Yet the project [https://github.com/kreier/jetson-car](https://github.com/kreier/jetson-car) stalled from 2020-2024.

![Jetson and car](https://github.com/kreier/jetson-car/blob/main/pic/2019_jetson_car.jpg?raw=true)

## OpenAI releases ChatGPT and the world takes note - November 2022

The underlying model GPT-3.5 was already released in March 2022. But in order to make it an assistant you can talk to some __finetuning__ had to be done. See blog post [https://openai.com/index/chatgpt/](https://openai.com/index/chatgpt/)

![Finetuning ChatGPT](https://github.com/kreier/nano-gpt/blob/main/docs/finetuning.jpg?raw=true)

And ever since the interface was available on the web with ChatGPT since November 2022 the world took note. It ran in news outlets and personal conversations worldwide and within two years NVIDIA became the most valuable company in the world.

## Challenges of self driving - 2024

It seemed like the combination of object detection and classification was the main problem to get self driving cars possible, together with latency. As it turns out, there is much more to do. Additinal sensors like LIDAR really help, even Tesla finally gave in. And here are some example videos to show whats possible and what is still challenging:

- [Why self-driving cars have stalled - It's Complicated](https://www.youtube.com/watch?v=4sCK-a33Nkk) by The Guardian September 2022
- [I RACED My Self-Driving RC Car](https://www.youtube.com/watch?v=zuyOdaQ2xuw) by Steven Gong in July 2023, taking part in the [F1 tenth](https://f1tenth.org/) race 2023
- [Sorry. Your Car Will Never Drive You Around.](https://www.youtube.com/watch?v=2DOd4RLNeT4) - from January 2024, self driving cars are 8times __less__ save than humans, see data at minute 15 and 11.3 deaths per 100 million miles for Tesla FSD versus 1.35 deaths for humans
- [Video First AI powered Race - A2RL Abu Dhabi](https://www.youtube.com/watch?v=feTxamTHQAA) - dissaster 2024-04-28 [https://a2rl.io/](https://a2rl.io/)

## Compute power needed for GPTs

- __GPT-1__ needed `8.6e19` FLOP
- __GPT-2__ needed `1.5e21` FLOP ([in 2019](https://x.com/karpathy/status/1811467135279104217) some [$100k](https://x.com/karpathy/status/1811488645175738409), possible [in July 2024 for $672](https://github.com/karpathy/llm.c/discussions/677))
- __GPT-3__ needed `3.1e23` FLOP
- __GPT-3.5__ undisclosed
- __GPT-4__ needed `2.1e25` FLOP estimated
- __3070 Ti__ running 24h: 2.3e13 FLOPS x 86400 = `1.9e18` FLOP - not even enough for GPT-1 (according to [test with OpenCL](https://github.com/kreier/benchmark/tree/main/gpu/opencl)), that would require 45 days!
- __nanoGPT__ estimated at `1.75e16` (see below) or 800 seconds on my 3070 Ti

__First run__ of `bigram.py` at [0:41:44](https://youtu.be/kCc8FmEb1nY?si=sS4_QPI_ic_WNMt0&t=2504) with 3000 iterations in just a few seconds

``` py
batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-2
```

Then introducing several improvements:

- Averaging past content with for loops
- Matrix multiply as weighted aggregation: the trick in self-attention
- Adding softmax
- [1:02:00](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3720s) THE CRUX OF THE VIDEO: version 4: self-attention

__Second run__ at [1:21:28](https://youtu.be/kCc8FmEb1nY?si=veYi24diWjjoPKMB&t=4888) with `train loss 2.39` and `val loss 2.41`. Still very fast, just a few seconds. Here are some parameters:

``` py
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

Memory won't be the limiting factor, as tested with `bigram.py` on both CPU and GPU. And the script does not explicitly uses TF32 ([only 10 bits fraction/significand/mantissa](https://www.exxactcorp.com/blog/hpc/what-is-fp64-fp32-fp16)) but FP32 (23 bits). That's three more bits for fraction than BF16 (7 bits), while all three (FP32, TF32, BF16) have 1 sign bit and 8 bit exponent. The FP32 power of [the A100](https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821) is 19.49 TFLOPS, so for 15 min = 900 seconds the needed compute is `1.75e16`. Having [21.75 TFLOPS FP32](https://www.techpowerup.com/gpu-specs/geforce-rtx-3070-ti.c3675) on my 3070 Ti the training run of 5000 iterations should take 800 seconds or 13 minutes.

__Update 2024/07/24__: The training run `python train.py config/train_shakespeare_char.py` actually needs just needs 27 seconds to compile, another 24 seconds from the first step to the first iteration. And with iteration times for around 45 ms all  5000 iterations to a final loss of 0.82 (step 5000: train loss 0.62, val loss 1.69) need 6:51 minutes or 411 seconds. GPU dedicated RAM increased from 0.6 GB to 2.7 GB, some 2.1 GB are needed for this exercise. Output from prepare:

``` sh
mk@i3:~/nanogpt$ python data/shakespeare_char/prepare.py
length of dataset in characters: 1,115,394
all the unique characters:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab size: 65
train has 1,003,854 tokens
val has 111,540 tokens
```

And training:

``` sh
mk@i3:~/nanogpt$ python train.py config/train_shakespeare_char.py
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000

tokens per iteration will be: 16,384
found vocab_size = 65 (inside data/shakespeare_char/meta.pkl)
Initializing a new model from scratch
number of parameters: 10.65M
num decayed parameter tensors: 26, with 10,740,096 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters
using fused AdamW: True

step 5000: train loss 0.6210, val loss 1.6979
iter 5000: loss 0.8202, time 7435.02ms, mfu 7.53%
```

__Update 2024/07/29:__ Andrej mentions that he uses a A100 GPU. In the video the time needed is 15 minutes, on [the website for nanoGPT](https://github.com/karpathy/nanoGPT) it is down to just 3 minutes. My 3070 Ti is only half as fast. But there is a cheaper option: in [Google Colab](https://colab.research.google.com/) you can use a GPU instance called T4. That's a [Nvidia Tesla T4](https://www.pny.com/en-eu/nvidia-t4) data center GPU with 16 GB RAM and [Turing Microarchitecture](https://en.wikipedia.org/wiki/Turing_(microarchitecture)) (CUDA 7.5 and therefore compatible!). It does not support `bf16` and the CUDA compiler Triton throws some errors, but you can just add two lines to the top of the `train.py` and it compiles and trains!

``` python
import torch._dynamo
torch._dynamo.config.suppress_errors = True
```

We don't get the impressive 130 TOPS INT8 performance, and only a fraction of the 8.1 TFLOPS FP32 performance since we share it. But the cycle times of 520 ms are not that bad, and after one hour (Google provides more than 2 hours runtime for free) the model is trained and ready to use! See my [Jupyter Notebook at colab.google](https://colab.research.google.com/drive/1h2trNGGP_WfG49ADUg6K32217CnTZgC2?usp=sharing) or here at Github.

## Energy efficiency

| year | Device         | TOPS | Watt | FP16   | FP32  | Price | TOPS/Watt |
|------|----------------|-----:|-----:|--------|-------|------:|:---------:|
| 2017 | TPU v2         |   45 |  280 |      - |     - |       | 0.16      |
| 2018 | T4             |  130 |   70 |  64800 |  8100 |   900 | 1.86      |
| 2019 | Coral TPU      |    8 |    4 |      - |     - |    40 | 2.00      |
| 2021 | 3070 Ti        |   22 |  290 |  21750 | 21750 |   500 | 0.07      |
| 2023 | L4             |  242 |   72 | 121000 | 30300 |  2500 | 3.36      |
| 2024 | Grayskull e75  |  221 |   75 |        |       |   600 | 2.95      |
| 2024 | Wormhole n300s | 466  | 300  |        |       |  1400 | 1.55      |
