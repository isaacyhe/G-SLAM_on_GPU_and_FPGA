#ifndef INV_H
#define INV_H

#ifdef __cplusplus
extern "C"{
#endif


void inv_cuda(double *result, double *mat, unsigned int dim);

//__global__ void kernel1(double *result);

//__global__ void kernel2(double *result, double *temp, int *j, int *h, double *ratio, unsigned int dim);

#ifdef __cplusplus
}
#endif
#endif
