#ifndef INV_H
#define INV_H

#ifdef __cplusplus
extern "C"{
#endif

void inv_cuda(float *result, float *mat, unsigned int dim);

void dot_cuda(float *result, float *a, unsigned int a_y, unsigned int a_x, float *b, unsigned int b_y, unsigned int b_x, unsigned int dim);

#ifdef __cplusplus
}
#endif
#endif
