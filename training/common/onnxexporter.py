import torch
import torch.onnx
import os

def onnx_exporter(model, model_path, output_path, in_dim, batch_size=1, max_value=36):
    """
    Export a PyTorch model to ONNX format.
    
    :param model: The PyTorch model to export.
    :param model_path: Path to the saved PyTorch model state dictionary file (.pth).
    :param output_path: Path to save the output ONNX model file (.onnx).
    :param in_dim: Input dimension of the model.
    :param out_dim: Output dimension of the model.
    :param batch_size: Batch size for the dummy input tensor.
    """
    
    # Load the trained model state dictionary
    if not os.path.exists(model_path):
        print(f"Error: Trained model file not found at {model_path}")
        return

    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded trained model state dictionary from {model_path}")
    except Exception as e:
        print(f"Error loading trained model state dictionary: {e}")
        return

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
            model,
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


