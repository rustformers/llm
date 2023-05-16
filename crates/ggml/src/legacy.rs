//! This module exposes legacy functionality of GGML that has been extracted
//! to help bridge versions.

/// Quantization version 0.
pub mod qnt0 {
    use crate::ElementType;

    macro_rules! generate_dequantization_function {
        ($rust_name:ident, $c_name:ident, $doc:literal) => {
            #[doc=$doc]
            pub fn $rust_name(row: &[u8], out: &mut [f32], row_size: usize) {
                assert_eq!(row_size, out.len());
                unsafe {
                    ggml_sys::$c_name(
                        row.as_ptr() as *const _,
                        out.as_mut_ptr(),
                        row_size.try_into().unwrap(),
                    )
                }
            }
        };
    }

    generate_dequantization_function!(
        dequantize_row_q4_0,
        qnt0_ggml_dequantize_row_q4_0,
        "Dequantizes a QNT0 q4_0 row to f32."
    );

    generate_dequantization_function!(
        dequantize_row_q4_1,
        qnt0_ggml_dequantize_row_q4_1,
        "Dequantizes a QNT0 q4_1 row to f32."
    );

    generate_dequantization_function!(
        dequantize_row_q4_2,
        qnt0_ggml_dequantize_row_q4_2,
        "Dequantizes a QNT0 q4_2 row to f32."
    );

    generate_dequantization_function!(
        dequantize_row_q5_0,
        qnt0_ggml_dequantize_row_q5_0,
        "Dequantizes a QNT0 q5_0 row to f32."
    );

    generate_dequantization_function!(
        dequantize_row_q5_1,
        qnt0_ggml_dequantize_row_q5_1,
        "Dequantizes a QNT0 q5_1 row to f32."
    );

    generate_dequantization_function!(
        dequantize_row_q8_0,
        qnt0_ggml_dequantize_row_q8_0,
        "Dequantizes a QNT0 q8_0 row to f32."
    );

    /// Dequantizes a QNT0 row to f32.
    pub fn dequantize_row(
        element_type: ElementType,
        row: &[u8],
        out: &mut [f32],
        row_size: usize,
    ) -> bool {
        match element_type {
            crate::Type::Q4_0 => {
                dequantize_row_q4_0(row, out, row_size);
                true
            }
            crate::Type::Q4_1 => {
                dequantize_row_q4_1(row, out, row_size);
                true
            }
            crate::Type::LegacyQ4_2 => {
                dequantize_row_q4_2(row, out, row_size);
                true
            }
            crate::Type::Q5_0 => {
                dequantize_row_q5_0(row, out, row_size);
                true
            }
            crate::Type::Q5_1 => {
                dequantize_row_q5_1(row, out, row_size);
                true
            }
            crate::Type::Q8_0 => {
                dequantize_row_q8_0(row, out, row_size);
                true
            }
            _ => false,
        }
    }
}
