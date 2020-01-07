A simple neural network implementation from scratch using Chapel language
-------------------------------------------------------------------------



Compiling with Linear Algebra
-----------------------------
Some of the linear algebra module procedures rely on the :mod:`BLAS` and
:mod:`LAPACK` modules.  If using routines that rely on these modules,
be sure to have a BLAS and LAPACK implementation available on your system.
See the :mod:`BLAS` and :mod:`LAPACK` documentation for further details.

To install blas in mac
-----------------
brew install openblas

Known Warnings
--------------
Compilation will generate warnings about incompatible pointer types,
which may be ignored.
Example: warning: implicit conversion from enumeration type 'Order_chpl' to
      different enumeration type 'enum CBLAS_ORDER' [-Wenum-conversion]

These warnings are due to the header files of OpenBLAS differing from the
reference C_BLAS prototypes for complex arguments by using ``float*`` and
``double*`` pointers, instead of ``void*`` pointers.

To Compile
----------
chpl -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lblas hello.chpl --fast

To Run
------
./hello --maxloop anynumber

./hello --maxloop 2

Thanks to:
https://github.com/jonasbostoen/simple-neural-network
https://hpc-carpentry.github.io/hpc-chapel/03-ranges-arrays/
https://github.com/chapel-lang/chapel
https://learnxinyminutes.com/docs/chapel/
https://chapel-lang.org/tutorials/Oct2018/02-BaseLang.pdf
https://chapel-lang.org/docs/primers/randomNumbers.html

