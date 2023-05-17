// The dequantization functions from ggerganov/ggml@effcfa62da543e71affe6c39b78d0064f0c5d71d,
// which was the last version to support the first quantization format (0).
//
// https://github.com/ggerganov/ggml/tree/effcfa62da543e71affe6c39b78d0064f0c5d71d

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef  __cplusplus
    // restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif
    void qnt0_ggml_dequantize_row_q4_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
    void qnt0_ggml_dequantize_row_q4_1(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
    void qnt0_ggml_dequantize_row_q4_2(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
    void qnt0_ggml_dequantize_row_q5_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
    void qnt0_ggml_dequantize_row_q5_1(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
    void qnt0_ggml_dequantize_row_q8_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

#ifdef  __cplusplus
}
#endif
