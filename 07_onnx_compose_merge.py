import onnx
from onnx import compose

# Load the models
model1 = onnx.load("preprocessing.onnx")
model2 = onnx.load("eva02_large_patch14_448.onnx")

# Merge the models
merged_model = compose.merge_models(
    model1,
    model2,
    io_map=[("output_preprocessing", "input")],
    prefix1="preprocessing_",
    prefix2="model_",
    doc_string="Merged preprocessing and eva02_large_patch14_448 model",
    producer_name="dickson.neoh@gmail.com using onnx compose",
)

# Save the merged model
onnx.save(merged_model, "merged_model_compose.onnx")
