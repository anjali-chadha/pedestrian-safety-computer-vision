### CPU vs GPU
* Deep Learning uses GPU. Why?
* Nvidia is a major player in this space.
* Cores:
    * CPU uses fewer cores but each core is much faster and much more capable; great at sequential tasks
    * GPU has more cores but each core is much slower and dumber; great at parallel tasks.
    * CPU - say 4 cores
    * GPU - say 1920 cores
* Memory:
    * CPU may has its own small cache. Mainly depends on the computer RAM.
    * GPU has its own RAM built into chip.
* TitanX - Nividia has 12 GB RAM, 3840 cores
* Matrix Multiplication is perfectly suited for GPU. Can parallelize the computations.
* CPU will do the same thing sequetially in each core.
* Convolution - similar story; GPU can parallelize the computation

## Programming GPUs
* CUDA (Nvidia); write C like code
* Difficult to write code on CUDA
* NVidia has released their own libraries - cuDNN, very optimized for Nvidia hardware
* OpenCL - similar to CUDA, but runs on everything.

## CPU/GPU communication
* Model is on GPU, but big dataset is on the harddrive or SSD
* May become a bottleneck in the training.
* Solutions -
    * Read all data into RAM
    * Use SSD instead of HDD
    * Use multiple CPU threads to prefetch data.


[YouTube link](https://www.youtube.com/watch?v=6SlgtELqOWc)

Keras, Tensorflow, Caffe, PyTorch, Theano.... (Too many options!!?)

When to use what?

## Frameworks
* Microsoft CNTK
* Google TF
* Google Keras
* Facebook Caffe2
* Facebook PyTorch
* Amazon MXNet
* Paddle Baidu

* Computational Graph for deep learning
* Neural Train Machines - Big graph
* With DL frameworks,
    * Easy to build computational graphs
    * Easy to compute gradients in computational graphs
 
 
 * Numpy can't run on GPU
 * Numpy can't compute gradients automatically
 
 Tensorflow -
 * Divide the computation into two stages:
    * Define the computational graph
    * Run the graph many times with the input data
 * Create placeholders - they are input to the graph.  - Not allocating any memory
 * Setting up the computatation using these placeholders.
 * Tensorflow Session to enter the data (generally numpy arrays)
 * session.run to execute the graph
 * For training the network - call session run multiple times
 * Forward pass - feeding in the weights (numpy arrays), copying the data between tensorflow and numpy arrays
 * Expensive to copy data between cpu and gpu
 * Solution to the above problem -
    * Rather than having w1 and w2 as placeholders. Use them as Variables
    * They will live inside the graph, and they should be initialised by the tensorflow.
    * The mutation of w1 and w2 should also be an operation
 

## Discussion
* Keras is high-level API built on Tensorflow (and can be used on top of Theano too).
* More user-friendly as compared to TF.
* Hmm.. then why will I ever use Tensorflow?
* Keras useful for rapid prototyping. Using keras, can build very simple or very complex nerual networks within a few minutes.
* Keras - pythonic, modularity
* Keras code is portable - can implement nerual network in Keras using Theano as a backend and then specidy the backend to run o TF, with no changes to code
* Tensorflow - use that to tinker with low-level details of neural network.
* TF offers more advanced operations as compared to Keras, and gives more control over network
* TF built around concept of Static Computational Graph. First, you define everything that is going to happen inside your framework, then you run it.
* PyTorch is Dynamic Computational Graph. Advantages -
  - Networks are modular. Can implement and debug each part separately.
  - Dynamic data structures inside network. Can define, change, execute nodes as you go.
  - In RNNs - input sequence length has to stay constant for static graphs. For instance, for a sentiment analysis model of English sentences
  we have to fix the sentence length to some maximum value and pad smaller sequences with zeros.

## Programming Details
* Both TF and Theano expects a 4-D tensor as input. There are some differences:
  - TF tensor format - (samples, rows, cols, **channels**)
  - Theano tensor format - (samples, **channels**, rows, cols)
  - To avoid ambiguity, while programming in Keras, we explicitly set the image dimension ordering to either 'tf' or 'th'
  
* Keras provide ability to **freeze** layers i.e we will not update the weights of the layer. This is useful when we are fine tuning a model. This is acheived by passing *trainable=False* parameter.
