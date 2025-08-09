import torch
import torch.onnx
import os
import torch.nn.functional as F

class ExportModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, decision_fn):
        super().__init__()
        self.model = model
        self.decision_fn = decision_fn

    def forward(self, x):
        logits = self.model(x)
        return self.decision_fn(logits, dim=-1)


def onnx_exporter(model, decision_fn, output_path, in_dim, batch_size=1, max_value=36):
    """
    Export a PyTorch model to ONNX format.
    
    :param model: The PyTorch model to export.
    :param output_path: Path to save the output ONNX model file (.onnx).
    :param in_dim: Input dimension of the model.
    :param out_dim: Output dimension of the model.
    :param batch_size: Batch size for the dummy input tensor.
    """
    model.eval()
    export_model = ExportModelWrapper(model, decision_fn)

    
    # Create dummy input tensors
    dummy_input = torch.randint(0, max_value, (batch_size, in_dim), dtype=torch.long)

    # Define input and output names for the ONNX graph
    input_names = ["state"]
    output_names = ["action"]

    # Define dynamic axes
    dynamic_axes = {
        "state": {0: "batch_size"},
        "action": {0: "batch_size"}
    }

    # Export the model to ONNX format
    print("Exporting model to ONNX format...")
    try:
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print(f"Successfully converted and saved ONNX model to {output_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")


