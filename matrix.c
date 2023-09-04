#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if(rows<1||cols<1)
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect values for row and col");
        return -1;
    }
    *mat = (matrix*)malloc(sizeof(matrix));
    if(mat==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "malloc mat failed!");
        return -1;
    }
    (*mat)->cols=cols;
    (*mat)->rows=rows;
    (*mat)->parent=NULL;
    (*mat)->ref_cnt=1;
    (*mat)->data = calloc(cols*rows,sizeof(double));
    if((*mat)->data==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "malloc data failed!");
        return -1;
    }
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if(rows<1||cols<1)
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect values for row and col");
        return -1;
    }
    *mat = (matrix*)malloc(sizeof(matrix));
    if(mat==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "malloc mat failed!");
        return -1;
    }
    (*mat)->cols=cols;
    (*mat)->rows=rows;
    (*mat)->parent=from;
    (*mat)->ref_cnt=1;
    matrix *curParent = (*mat)->parent;
    while(curParent!=NULL)
    {
        curParent->ref_cnt++;
        curParent=curParent->parent;
    }
    (*mat)->data=from->data+offset;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if(mat == NULL)
    {
        return;
    }
    mat->ref_cnt--;
    deallocate_matrix(mat->parent);
    if(mat->ref_cnt==0)
    {
        if(mat->parent==NULL)
        {
            free(mat->data);
        }
        free(mat);
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    return mat->data[row*mat->cols+col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    mat->data[row*mat->cols+col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    int rows = mat->rows;
    int cols = mat->cols;
    double *data = mat->data;

    __m256d vals = _mm256_set1_pd(val);

    #pragma omp parallel for collapse(2)
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols/4*4; j+=4)
        {
            _mm256_storeu_pd(data+i*cols+j,vals);
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i=0; i<rows; i++)
    {
        for(int j=cols/4*4; j<cols; j++)
        {
            data[i*cols+j]=val;
        }
    }

}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    int rows = result->rows;
    int cols = result->cols;

    double*A = mat1->data;
    double*B = mat2->data;
    double*C = result->data;

    #pragma omp parallel for collapse(2)
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols/4*4; j+=4)
        {
            int offset = i*cols+j;
            __m256d a = _mm256_loadu_pd(A + offset);
            __m256d b = _mm256_loadu_pd(B + offset);
            _mm256_storeu_pd(C + offset,_mm256_add_pd(a,b));
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i=0; i<rows; i++)
    {
        for(int j=cols/4*4; j<cols; j++)
        {
            int offset = i*cols+j;
            C[offset] = A[offset] + B[offset];
        }
    }

    return 0;    
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    int rows = result->rows;
    int cols = result->cols;

    double*A = mat1->data;
    double*B = mat2->data;
    double*C = result->data;

    #pragma omp parallel for collapse(2)
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols/4*4; j+=4)
        {
            int offset = i*cols+j;
            __m256d a = _mm256_loadu_pd(A + offset);
            __m256d b = _mm256_loadu_pd(B + offset);
            _mm256_storeu_pd(C + offset,_mm256_sub_pd(a,b));
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i=0; i<rows; i++)
    {
        for(int j=cols/4*4; j<cols; j++)
        {
            int offset = i*cols+j;
            C[offset] = A[offset] - B[offset];
        }
    }

    return 0;   
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */

#define UNROLL 8
#define BLOCKSIZE 32
#define BREAKSIZE 256

int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
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

    //break the mats
    memset(C, 0, m1r*m2c*sizeof(double));

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
        }
        
    }

    #pragma omp parallel for collapse(2)
    for(int i=0; i<m1r/BLOCKSIZE*BLOCKSIZE; i+=BLOCKSIZE)
    {
        for(int j=m2c/BLOCKSIZE*BLOCKSIZE; j<m2c; j++)
        {   //block k
            for(int k=0; k<m1c/BLOCKSIZE*BLOCKSIZE; k+=BLOCKSIZE)
            {
                for(int bi=i; bi<i+BLOCKSIZE; bi++)
                {
                    for(int bk=k; bk<k+BLOCKSIZE;bk++)
                    {
                        C[bi*m2c+j]+=A[bi*m1c+bk]*B[bk*m2c+j];
                    }
                }
            }

            //out k
            for(int ok=m1c/BLOCKSIZE*BLOCKSIZE; ok<m1c; ok++)
            {
                for(int bi=i; bi<i+BLOCKSIZE; bi++)
                {
                    C[bi*m2c+j]+=A[i*m1c+ok]*B[ok*m2c+j];
                }
            }
        }
    }


    #pragma omp parallel for collapse(2)
    for(int i=m1r/BLOCKSIZE*BLOCKSIZE; i<m1r;i++)
    {
        for(int j=0; j<m2c/BLOCKSIZE*BLOCKSIZE; j+=BLOCKSIZE)
        {
            for(int k=0; k<m1c/BLOCKSIZE*BLOCKSIZE; k+=BLOCKSIZE)
            {
            _mm_prefetch(C + i*m2c+j,_MM_HINT_NTA);
            for(int bj=j;bj<j+BLOCKSIZE; bj+=step)
            {
                __m256d c[UNROLL];
                for(int x=0; x<UNROLL; x++)
                {
                    c[x]=_mm256_loadu_pd(C+i*m2c+bj+x*4);
                }
                for(int bk=k; bk<k+BLOCKSIZE; bk++)
                {
                    __m256d a = _mm256_broadcast_sd(A+i*m1c+bk);
                    for(int x=0;x<UNROLL;x++)
                    {
                        __m256d b = _mm256_loadu_pd(B+bk*m2c+bj+4*x);
                        c[x]=_mm256_fmadd_pd(a,b,c[x]);
                    }
                }
                for(int x=0;x<UNROLL;x++)
                {
                    _mm256_storeu_pd(C+i*m2c+bj+x*4,c[x]);
                }
            }
            }
            for(int ok=m2r/BLOCKSIZE*BLOCKSIZE;ok<m2r;ok++)
            {
                _mm_prefetch(C + i*m2c+j,_MM_HINT_NTA);
                for(int bj=j;bj<j+BLOCKSIZE;bj+=step)
                {
                    __m256d c[UNROLL];
                    for(int x=0;x<UNROLL;x++)
                    {
                        c[x]=_mm256_loadu_pd(C+i*m2c+bj+x*4);
                    }
                    __m256d a = _mm256_broadcast_sd(A+i*m1c+ok);
                    for(int x=0;x<UNROLL;x++)
                    {
                        __m256d b = _mm256_loadu_pd(B+ok*m2c+bj+4*x);
                        c[x]=_mm256_fmadd_pd(a,b,c[x]);
                    }
                    for(int x=0;x<UNROLL;x++)
                    {
                        _mm256_storeu_pd(C+i*m2c+bj+x*4,c[x]);
                    }
                }
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i=m1r/BLOCKSIZE*BLOCKSIZE;i<m1r;i++)
    {
        for(int j=m2c/BLOCKSIZE*BLOCKSIZE;j<m2r;j++)
        {
            for(int k=0;k<m2r/BLOCKSIZE*BLOCKSIZE;k+=BLOCKSIZE)
            {
                for(int bk=k;bk<k+BLOCKSIZE;bk++)
                {
                    C[i*m2c+j]+=A[i*m1c+bk]*B[bk*m2c+j];
                }
            }
            for(int ok=m2r/BLOCKSIZE*BLOCKSIZE;ok<m2r;ok++)
            {
                C[i*m2c+j]+=A[i*m1c+ok]*B[ok*m2c+j];
            }
        }
    }

    return 0;
}




/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
 int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    matrix** tt = malloc(sizeof(matrix*));
    allocate_matrix(tt,mat->rows,mat->cols);
    matrix* t = *tt;

    matrix** pp = malloc(sizeof(matrix*));
    allocate_matrix(pp,mat->rows,mat->cols);
    matrix* p = *pp;

    matrix* x = result;

    memset(result->data,0,result->cols*result->rows*sizeof(double));

    #pragma omp parallel for
    for(int i=0; i<mat->rows;i++)
    {
        (result->data)[i+i*result->cols]=1.0;
    }

    memcpy(p->data,mat->data,mat->cols*mat->rows*sizeof(double));

    while(pow)
    {
        if(pow%2!=0)
        {
            mul_matrix(t,result,p);
            //memcpy(result->data,t->data,mat->cols*mat->rows*sizeof(double));
            matrix* temp = result;
            result = t;
            t = temp;
        }
        mul_matrix(t,p,p);
        //memcpy(p->data,t->data,mat->cols*mat->rows*sizeof(double));
        matrix* temp = p;
        p = t;
        t = temp;
        pow/=2;
    }

    if(result!=x)
    memcpy(x->data,result->data,mat->cols*mat->rows*sizeof(double));

    deallocate_matrix(*pp);
    free(pp);

    deallocate_matrix(*tt);
    free(tt);

    return 0;

 }

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    int rows = result->rows;
    int cols = result->cols;

    double* A = mat->data;
    double* B = result->data;

    #pragma omp parallel for
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols/4*4; j+=4)
        {
            __m256d a =_mm256_loadu_pd(A + i*cols +j);
            _mm256_storeu_pd(B+i*cols +j,_mm256_xor_pd(a,_mm256_set1_pd(-0.0)));
        }

        for(int j=cols/4*4; j<cols ;j++)
        {
            B[i*cols+j] = -A[i*cols+j];
        }
    }

    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    int rows = result->rows;
    int cols = result->cols;

    double* A = mat->data;
    double* B = result->data;

    const __m256d b = _mm256_set1_pd(-1.0);

    #pragma omp parallel for
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols/4*4; j+=4)
        {
            __m256d a =_mm256_loadu_pd(A + i*cols +j);
            _mm256_storeu_pd(B+i*cols +j,_mm256_max_pd(a,_mm256_mul_pd(a,b)));
        }

        for(int j=cols/4*4; j<cols ;j++)
        {
            B[i*cols+j] = fabs(A[i*cols+j]);
        }
    }

    return 0;
}

