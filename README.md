# numc

**本文旨在帮助想要自学并完成cs61c proj4（su20）的朋友，同时记录了我对这个proj的理解，同时我参考了[别人的仓库](https://github.com/ashvindhawan/NumRec.git)**

*同时由于我个人无法弄好dumbpy包所以无法完成pytest测试，所以我只能在test1.py程序中简单测试了一下，当然我进行了所有的cunit测试（由于我修改了部分测试代码，并没有设置其正确结果，所以你会看到有没有通过的测试），如果你有办法进行pytest，请告诉我！*

Here's what I did in project 4:
# 准备工作
## conda环境

我是在Ubuntu下配置的python3.10环境，python版本无所谓因为你不好用他的dumbpy包。
安装好后修改makefile
```shell
PYTHON = -L/usr/lib/python3.10 -I/usr/include/python3.10 -lpython3.10
```
## cunit测试环境
推荐你在[gitlab](https://gitlab.com/cunity/cunit)上下载，官网下载的config文件有问题无法正确安装，下好后按说明用cmake安装
```shell
mkdir local-build
cd local-build
cmake ..
cmake --build .
```
安装好后修改makefile
```shell
CUNIT = -L/home/alpha/cunit/CUnit/local-build -I/home/alpha/cunit/CUnit -lcunit
```
**记得改成你自己的路径**

gcc编译命令解释：`-L`指定库的搜索路径，并使用 `-l `指定库名,`-I`指定头文件的包含路径。

##检查你安装好了没有
我copy了一个简单的cunit程序，并包含了`Python.h`。
```c
#include<Python.h>
#include "CUnit/CUnitCI.h"

/* Test that one equals one */
static void test_simple_pass1(void) {
    CU_ASSERT_FATAL(1 == 1);
}

CUNIT_CI_RUN("my-suite",
             CUNIT_CI_TEST(test_simple_pass1));
```
并在makefile中加上了
```Makefile
hello:
	rm -f hello
	$(CC) $(CFLAGS) hello.c -o hello $(LDFLAGS) $(CUNIT) $(PYTHON)
	./hello
```
如果你能`make hello`通过这个测试，说明你已经配置好了环境，可以开始project4了！

# 矩阵乘法及加速工作讲解

如果你还不知道矩阵操作的加速方法，以及为什么要这么做，快把lab做完吧。
```
#define UNROLL 8 //循环展开次数，就是一次循环进行多少次操作
#define BLOCKSIZE 32 //矩阵分块时每一块的大小
#define BREAKSIZE 256 //当矩阵大于多少时进行分块处理
```
## 1.不分块时
```c
    int m1r = mat1->rows, m1c = mat1->cols, m2r = mat2->rows, m2c =mat2->cols;
    double *A = mat1->data;
    double *B = mat2->data;
    double *C = result->data;
    int step = 4*UNROLL;

    if(m1r < BREAKSIZE ||m1c < BREAKSIZE ||m2r < BREAKSIZE ||m2c < BREAKSIZE)
    {
        
        for(int i =0; i<m1r; i++)
        {
            for(int j=0; j<m2c/step*step; j+=step)
            {
                __m256d c[UNROLL];
                for(int x=0; x< UNROLL; x++)
                {
                    c[x] = _mm256_setzero_pd();
                }

                for(int k=0; k<m1c; k++)
                {
                    __m256d a = _mm256_broadcast_sd(A + i*m1c +k);
                    for(int x=0; x<UNROLL; x++)
                    {
                        __m256d b = _mm256_loadu_pd(B + k*m2c +(j+4*x));
                        c[x]= _mm256_fmadd_pd(a,b,c[x]);
                    }
                }

                for(int x=0; x<UNROLL;x++)
                {
                    _mm256_storeu_pd(C + i*m2c + (j+4*x),c[x]);
                }
            }
        }
        //对于j上面是一次处理一个step，下面是处理不到一个step的部分
        for(int i=0; i<m1r; i++)
        {
            for(int j = m2c/step*step; j<m2c; j++)
            {
                double sum=0.0;
                for(int k = 0; k<m1c; k++)
                {
                    sum += A[i*m1c + k]*B[k*m2c +j];
                }
                C[i*m2c + j]=sum;
            }
        }

        return 0;
    }

```
这里不过是对于普通的三次循环在j上实现了一次处理32位的操作，

## 分块时
你可以想象一下我们用32*32的块将一个大矩阵分成小块，但是如果矩阵的长宽都不是32的倍数，矩阵就可以看成4块
| 1 | 2 |
|---|---|
| 3 | 4 |

1是能切好块的
2是cols剩余的
3是rows剩余的
4是cols，rows剩余的

同理我们可以将mat1，mat2，result都看成这样2*2的矩阵
所以
mat1:
| a11 | a12 |
|----|----|
| a21 | a22 |

mat2:
| b11 | b12 |
|----|----|
| b21 | b22 |

result:
| a11*b11+a12*b21 | a11*b12+a12*b22 |
|------------------|------------------|
| a21*b11+a22*b21 | a21*b12+a22*b22 |


下面是这个块的代码
|  a11*b11+a12*b21 |
|---|
```c
 #pragma omp parallel for collapse(2)
    for(int i=0; i<m1r/BLOCKSIZE*BLOCKSIZE; i+=BLOCKSIZE)
    {
        for(int j=0; j<m2c/BLOCKSIZE*BLOCKSIZE; j+=BLOCKSIZE)
        {   //block k
            for(int k=0; k<m1c/BLOCKSIZE*BLOCKSIZE; k+=BLOCKSIZE)
            {
                for(int bi=i; bi<i+BLOCKSIZE; bi++)
                {
                    _mm_prefetch(C+bi*m2c+j,_MM_HINT_NTA);
                    for(int bj=j;bj<j+BLOCKSIZE;bj+=step)
                    {
                        __m256d c[UNROLL];
                    for(int x=0; x< UNROLL; x++)
                    {
                        c[x] = _mm256_loadu_pd(C + bi*m2c +bj +x*4);
                    }

                    for(int bk=k; bk<k+BLOCKSIZE; bk++)
                    {
                        __m256d a = _mm256_broadcast_sd(A + bi*m1c +bk);
                        for(int x=0; x<UNROLL; x++)
                        {
                            __m256d b = _mm256_loadu_pd(B + bk*m2c +(bj+4*x));
                            c[x]= _mm256_fmadd_pd(a,b,c[x]);
                        }
                    }

                    for(int x=0; x<UNROLL;x++)
                    {
                        _mm256_storeu_pd(C + bi*m2c + (bj+4*x),c[x]);
                    }  
                    }
                }
            }
            //上面是a11*b11的代码
            //out k
            for(int ok=m1c/BLOCKSIZE*BLOCKSIZE; ok<m1c; ok++)
            {
                for(int bi=i; bi<i+BLOCKSIZE; bi++)
                {
                    _mm_prefetch(C+bi*m2c+j,_MM_HINT_NTA);
                    for(int bj=j;bj<j+BLOCKSIZE;bj+=step)
                    {
                        __m256d c[UNROLL];
                    for(int x=0; x< UNROLL; x++)
                    {
                        c[x] = _mm256_loadu_pd(C + bi*m2c +bj +x*4);
                    }

                    __m256d a = _mm256_broadcast_sd(A + bi*m1c +ok);
                        for(int x=0; x<UNROLL; x++)
                        {
                            __m256d b = _mm256_loadu_pd(B + ok*m2c +(bj+4*x));
                            c[x]= _mm256_fmadd_pd(a,b,c[x]);
                        }
                    for(int x=0; x<UNROLL;x++)
                    {
                        _mm256_storeu_pd(C + bi*m2c + (bj+4*x),c[x]);
                    }  
                    }
                }
            }
            //这里是a12*b21的代码
        }
        
    }
```
说明：`#pragma omp parallel for collapse(2)`是对接下来两层循环进行多线程化，`collapse`命令可以让线程负载均衡，之并行化两层循环，是为了让每个线程都有合适的任务量，充分利用每个线程。

后面的3个块也是相同的方法就不重复说明了。

# 矩阵的多次方计算
对于一个数的pow次方我们可以这样计算：
```c
int pow_num(int x,int pow)
{
    int result=1;
    int temp=x;
    while(pow)
    {
        if(pow%2==1)
        {
            result=result*temp;
        }
        temp=temp*temp;
        pow/=2;
    }
    return result;
}
```

        

-