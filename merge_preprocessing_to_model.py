from snc4onnx import combine

combined = combine(
    srcop_destop=["output_prep", "input"],
    input_onnx_file_paths=[
        "01_prep_1_3_448_448.onnx",
        "eva02_large_patch14_448.onnx",
    ],
    op_prefixes_after_merging=["prep", "model"],
)


combined.save("merged_preprocessing_to_model.onnx")


# snc4onnx --input_onnx_file_paths 01_prep_1_3_448_448.onnx eva02_large_patch14_448.onnx --output_onnx_file_path merged.onnx --srcop_destop output_prep input --op_prefixes_after_merging init next
