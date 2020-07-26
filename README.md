# A simple neural network implementation from scratch using Chapel language
-------------------------------------------------------------------------



### Dataset
--------

Download the mnist dataset as csv from https://github.com/pjreddie/mnist-csv-png 



### Compiling with Linear Algebra
-----------------------------
Some of the linear algebra module procedures rely on the :mod:`BLAS` and
:mod:`LAPACK` modules.  If using routines that rely on these modules,
be sure to have a BLAS and LAPACK implementation available on your system.
See the :mod:`BLAS` and :mod:`LAPACK` documentation for further details.



### To install blas in mac
-----------------
brew install openblas



### Ignore Known Warnings
--------------
Compilation will generate warnings about incompatible pointer types,
for each "dot" or matrix multiplication, which may be ignored.

Example: warning: implicit conversion from enumeration type 'Order_chpl' to
      different enumeration type 'enum CBLAS_ORDER' [-Wenum-conversion]

In file included from /var/folders/69/k19bvb4x6pq9mqlm6q_t8my80000gn/T//chpl-kaushikvelusamy-872.deleteme/_main.c:79:
/var/folders/69/k19bvb4x6pq9mqlm6q_t8my80000gn/T//chpl-kaushikvelusamy-872.deleteme/BLAS.c:271:13: warning: 
      implicit conversion from enumeration type 'Order_chpl' to
      different enumeration type 'enum CBLAS_ORDER'
      [-Wenum-conversion]
cblas_dgemm(order_chpl, opA_chpl, opB_chpl, call_tmp_chpl124...
~~~~~~~~~~~ ^~~~~~~~~~

These warnings are due to the header files of OpenBLAS differing from the
reference C_BLAS prototypes for complex arguments by using ``float*`` and
``double*`` pointers, instead of ``void*`` pointers.



### To Compile
----------

cd mnist_neural_nets

chpl -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lblas nn.chpl --fast  -M ../lib/

or

 ~/chapel/chapel-1.20.0/bin/linux_ppc_le64-ppc64le/chpl -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lblas chapels_nn.chpl --fast  -M lib/


Add/Remove --fast 

This is a compiler
meta-flag that turns off several execution-time correctness checks
(bounds checks, NULL pointer checks, etc.) and turns on C-level
optimizations.  See the 'chpl' man page for details.


### To Run
------

./nn


or

./nn --train_file "FILE_PATH" --test_file "FILE_PATH" --learn_rate "VALUE BETWEEN 0.01-0.10" --training_epochs_iterations "VALUE BETWEEN 100-1000" --layer1_neurons "INT_VALUE" --layer2_neurons "INT_VALUE" --layer3_neurons "INT_VALUE"

When you run just ./nn the default values are

config const train_input_file = "dataset/xtrain.csv";
config const train_output_file = "dataset/ytrain.csv";
config const test_input_file  = "dataset/xval.csv";
config const test_output_file = "dataset/yval.csv";
config const learn_rate : real = 0.5;
config const training_epochs_iterations : int = 230;
const digits_range      = 0..9;
const pixels_per_line   = 1..64;                // 8 X 8 pixels
config const layer1_neurons: int         = 64;
config const layer2_neurons: int         = 128;
config const layer3_neurons: int         = 128;


### To Run pythons nn
---------------------

python3 pythons_nn.py


### My ANN Network Sample
-----------------

Image Reference from :
https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7

Reference-style: 
![alt text][logo]

[logo]: https://miro.medium.com/max/840/1*o7VCg1WILHZMVoPAALKWYg.png "My sample network"


### Thanks to:
---------
https://github.com/jonasbostoen/simple-neural-network

https://hpc-carpentry.github.io/hpc-chapel/03-ranges-arrays/

https://github.com/chapel-lang/chapel

https://learnxinyminutes.com/docs/chapel/

https://chapel-lang.org/tutorials/Oct2018/02-BaseLang.pdf

https://chapel-lang.org/docs/primers/randomNumbers.html

https://github.com/pjreddie/mnist-csv-png

https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7


