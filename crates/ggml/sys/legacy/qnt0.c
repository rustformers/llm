#include "qnt0.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#ifdef __ARM_NEON
    // we use the built-in 16-bit float type
    typedef __fp16 ggml_fp16_t;
#else
    typedef uint16_t ggml_fp16_t;
#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#ifndef __SSE3__
#define __SSE3__
#endif
#endif

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#define GGML_COMPUTE_FP16_TO_FP32(x) ((float) (x))
#define GGML_COMPUTE_FP32_TO_FP16(x) (x)

#define GGML_FP16_TO_FP32(x) ((float) (x))
#define GGML_FP32_TO_FP16(x) (x)

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#include <immintrin.h>
#endif
#endif

#ifdef __F16C__

#ifdef _MSC_VER
#define GGML_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define GGML_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define GGML_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

#elif defined(__POWER9_VECTOR__)

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
/* the inline asm below is about 12% faster than the lookup method */
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    register float f;
    register double d;
    __asm__(
        "mtfprd %0,%2\n"
        "xscvhpdp %0,%0\n"
        "frsp %1,%0\n" :
        /* temp */ "=d"(d),
        /* out */  "=f"(f):
        /* in */   "r"(h));
    return f;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
    register double d;
    register ggml_fp16_t r;
    __asm__( /* xscvdphp can work on double or single precision */
        "xscvdphp %0,%2\n"
        "mffprd %1,%0\n" :
        /* temp */ "=d"(d),
        /* out */  "=r"(r):
        /* in */   "f"(f));
    return r;
}

#else

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#endif // __F16C__

#endif // __ARM_NEON

//
// quantization
//

#if __AVX__ || __AVX2__ || __AVX512F__
// Unpack 16 4-bit fields into 16 bytes
// The output vector contains 16 bytes, each one in [ 0 .. 15 ] interval
static inline __m128i bytes_from_nibbles_16(const uint8_t * rsi)
{
    // Load 8 bytes from memory
    __m128i tmp = _mm_loadl_epi64( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m128i bytes = _mm_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m128i lowMask = _mm_set1_epi8( 0xF );
    __m128i high = _mm_andnot_si128( lowMask, bytes );
    __m128i low = _mm_and_si128( lowMask, bytes );
    high = _mm_slli_epi16( high, 4 );
    bytes = _mm_or_si128( low, high );
    return bytes;
}

#if __AVX2__ || __AVX512F__
// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    __m256i high = _mm256_andnot_si256( lowMask, bytes );
    __m256i low = _mm256_and_si256( lowMask, bytes );
    high = _mm256_slli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );
    return bytes;
}
#endif
#endif // __AVX__ || __AVX2__ || __AVX512F__

#if __ARM_NEON

#if !defined(__aarch64__)

int8x8_t vzip1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    return res;
}

int8x8_t vzip2_s8(int8x8_t a, int8x8_t b) {
    int8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    return res;
}

uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    return res;
}

uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    return res;
}

#endif
#endif


#define QK4_0 32
typedef struct {
    float   d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    float   d;          // delta
    float   m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(float) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK4_2 16
typedef struct {
    ggml_fp16_t d;         // delta
    uint8_t qs[QK4_2 / 2]; // nibbles / quants
} block_q4_2;
static_assert(sizeof(block_q4_2) == sizeof(ggml_fp16_t) + QK4_2 / 2, "wrong q4_2 block size/padding");

#define QK5_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    ggml_fp16_t d;         // delta
    ggml_fp16_t m;         // min
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    float   d;          // delta
    int8_t  qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(float) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    float   d;          // delta
    float   s0;         // d * sum(qs[i]) low
    float   s1;         // d * sum(qs[i]) high
    int8_t  qs[QK8_1];  // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 3*sizeof(float) + QK8_1, "wrong q8_1 block size/padding");

void qnt0_ggml_dequantize_row_q4_0(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    const block_q4_0 * restrict x = vx;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        // scale factor
        const __m256 d_v = _mm256_broadcast_ss(&x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 32) {
            // Load 32x4-bit integers into 32x8-bit integers
            __m256i vx8 = bytes_from_nibbles_32(pp+l/2);

            // Subtract 8 from the integers
            vx8 = _mm256_sub_epi8(vx8, _mm256_set1_epi8(8));

            // Convert to 16-bit int
            const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
            const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

            // Convert to 32-bit int -> float 32
            const __m256 vf[4] = {
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1)))
            };

            // Scale and store
            for (int j = 0; j < 4; j++) {
                const __m256 result = _mm256_mul_ps(vf[j], d_v);
                _mm256_storeu_ps(y + i * QK4_0 + l + j*8, result);
            }
        }
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        const float32x4_t vd = vdupq_n_f32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 16) {
            // Load 16x4-bit integers into 8x8-bit integers
            const uint8x8_t v8 = vld1_u8(pp + l/2);

            // Expand 4-bit qs to 8-bit bytes
            const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0F));
            const uint8x8_t v1 = vshr_n_u8(v8, 4);

            // Convert to signed 8-bit integers
            const int8x8_t vs_0 = vreinterpret_s8_u8(v0);
            const int8x8_t vs_1 = vreinterpret_s8_u8(v1);

            // Subtract 8 from each byte
            const int8x8_t vb_0 = vsub_s8(vs_0, vdup_n_s8(8));
            const int8x8_t vb_1 = vsub_s8(vs_1, vdup_n_s8(8));

            // Interleave and combine
            const int8x8_t vx_0 = vzip1_s8(vb_0, vb_1);
            const int8x8_t vx_1 = vzip2_s8(vb_0, vb_1);

            const int8x16_t vq = vcombine_s8(vx_0, vx_1);

            // convert to 2x int16x8_t
            const int16x8_t vi_0 = vmovl_s8(vget_low_s8 (vq));
            const int16x8_t vi_1 = vmovl_s8(vget_high_s8(vq));

            // convert to 4x float32x4_t
            const float32x4_t vf_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_0)));
            const float32x4_t vf_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_0)));
            const float32x4_t vf_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_1)));
            const float32x4_t vf_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_1)));

            // Multiply by d
            const float32x4_t r0 = vmulq_f32(vf_0, vd);
            const float32x4_t r1 = vmulq_f32(vf_1, vd);
            const float32x4_t r2 = vmulq_f32(vf_2, vd);
            const float32x4_t r3 = vmulq_f32(vf_3, vd);

            // Store
            vst1q_f32(y + i*QK4_0 + l +  0, r0);
            vst1q_f32(y + i*QK4_0 + l +  4, r1);
            vst1q_f32(y + i*QK4_0 + l +  8, r2);
            vst1q_f32(y + i*QK4_0 + l + 12, r3);
        }
    }
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0x0F;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            //printf("d = %f, vi = %d, vi0 = %d, vi1 = %d, v0 = %f, v1 = %f\n", d, vi, vi0, vi1, v0, v1);

            y[i*QK4_0 + l + 0] = v0;
            y[i*QK4_0 + l + 1] = v1;

            assert(!isnan(y[i*QK4_0 + l + 0]));
            assert(!isnan(y[i*QK4_0 + l + 1]));
        }
    }
#endif
}

void qnt0_ggml_dequantize_row_q4_1(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    const block_q4_1 * restrict x = vx;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        const __m256 d_v = _mm256_broadcast_ss(&x[i].d);
        const __m256 d_m = _mm256_broadcast_ss(&x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 32) {
            // Load 32x4-bit integers into 32x8-bit integers
            __m256i vx8 = bytes_from_nibbles_32(pp+l/2);

            // Convert to 16-bit int
            const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
            const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

            // Convert to 32-bit int -> float 32
            const __m256 vf[4] = {
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1)))
            };

            // Scale, add m and store
            for (int j = 0; j < 4; j++) {
                const __m256 result = _mm256_add_ps(_mm256_mul_ps(vf[j], d_v), d_m);
                _mm256_storeu_ps(y + i * QK4_1 + l + j*8, result);
            }
        }
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        const float32x4_t vd = vdupq_n_f32(x[i].d);
        const float32x4_t vm = vdupq_n_f32(x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 16) {
            // Load 16x4-bit integers into 8x8-bit integers
            const uint8x8_t v8 = vld1_u8(pp + l/2);

            // Expand 4-bit qs to 8-bit bytes
            const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0F));
            const uint8x8_t v1 = vshr_n_u8(v8, 4);

            // Interleave and combine
            const uint8x8_t vx_0 = vzip1_u8(v0, v1);
            const uint8x8_t vx_1 = vzip2_u8(v0, v1);

            const uint8x16_t vq = vcombine_u8(vx_0, vx_1);

            // convert to 2x uint16x8_t
            const uint16x8_t vi_0 = vmovl_u8(vget_low_u8 (vq));
            const uint16x8_t vi_1 = vmovl_u8(vget_high_u8(vq));

            // convert to 4x float32x4_t
            const float32x4_t vf_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vi_0)));
            const float32x4_t vf_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vi_0)));
            const float32x4_t vf_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vi_1)));
            const float32x4_t vf_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vi_1)));

            // multiply by d and add m
            const float32x4_t r0 = vmlaq_f32(vm, vf_0, vd);
            const float32x4_t r1 = vmlaq_f32(vm, vf_1, vd);
            const float32x4_t r2 = vmlaq_f32(vm, vf_2, vd);
            const float32x4_t r3 = vmlaq_f32(vm, vf_3, vd);

            // Store
            vst1q_f32(y + i*QK4_1 + l +  0, r0);
            vst1q_f32(y + i*QK4_1 + l +  4, r1);
            vst1q_f32(y + i*QK4_1 + l +  8, r2);
            vst1q_f32(y + i*QK4_1 + l + 12, r3);
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;
        const float m = x[i].m;

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0x0F;
            const int8_t vi1 = vi >> 4;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK4_1 + l + 0] = v0;
            y[i*QK4_1 + l + 1] = v1;

            assert(!isnan(y[i*QK4_1 + l + 0]));
            assert(!isnan(y[i*QK4_1 + l + 1]));
        }
    }
#endif
}

void qnt0_ggml_dequantize_row_q4_2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_2 == 0);
    const int nb = k / QK4_2;

    const block_q4_2 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_2; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0x0F;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            y[i*QK4_2 + l + 0] = v0;
            y[i*QK4_2 + l + 1] = v1;

            assert(!isnan(y[i*QK4_2 + l + 0]));
            assert(!isnan(y[i*QK4_2 + l + 1]));
        }
    }
}

void qnt0_ggml_dequantize_row_q5_0(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK5_0 == 0);
    const int nb = k / QK5_0;

    const block_q5_0 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int l = 0; l < QK5_0; l += 2) {
            const uint8_t vi = pp[l/2];

            // extract the 5-th bit from qh
            const uint8_t vh0 = ((qh & (1u << (l + 0))) >> (l + 0)) << 4;
            const uint8_t vh1 = ((qh & (1u << (l + 1))) >> (l + 1)) << 4;

            const int8_t vi0 = (vi & 0x0F) | vh0;
            const int8_t vi1 = (vi >>   4) | vh1;

            const float v0 = (vi0 - 16)*d;
            const float v1 = (vi1 - 16)*d;

            y[i*QK5_0 + l + 0] = v0;
            y[i*QK5_0 + l + 1] = v1;

            assert(!isnan(y[i*QK5_0 + l + 0]));
            assert(!isnan(y[i*QK5_0 + l + 1]));
        }
    }
}

void qnt0_ggml_dequantize_row_q5_1(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK5_1 == 0);
    const int nb = k / QK5_1;

    const block_q5_1 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int l = 0; l < QK5_1; l += 2) {
            const uint8_t vi = pp[l/2];

            // extract the 5-th bit from qh
            const uint8_t vh0 = ((qh & (1u << (l + 0))) >> (l + 0)) << 4;
            const uint8_t vh1 = ((qh & (1u << (l + 1))) >> (l + 1)) << 4;

            const uint8_t vi0 = (vi & 0x0F) | vh0;
            const uint8_t vi1 = (vi >>   4) | vh1;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK5_1 + l + 0] = v0;
            y[i*QK5_1 + l + 1] = v1;

            assert(!isnan(y[i*QK5_1 + l + 0]));
            assert(!isnan(y[i*QK5_1 + l + 1]));
        }
    }
}

void qnt0_ggml_dequantize_row_q8_0(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    const block_q8_0 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        const int8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK8_0; ++l) {
            y[i*QK8_0 + l] = pp[l]*d;
        }
    }
}
