import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

# Option 1: Copy the JassDNN class definition here
class JassDNN(nn.Module):
    """
    Deep Neural Network for Jass policy prediction using one-hot encoded inputs.
    (Definition for ONNX export script)
    """
    def __init__(self, num_cards=71, trump_dim=7, hidden_sizes=[512, 256, 128]):
        super().__init__()
        self.num_cards = num_cards
        total_input = 71 * num_cards + trump_dim

        layers = []
        in_dim = total_input
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, 9)

    def forward(self, state_idx, trump_onehot):
        # Cast state indices to float so ONNX Clip/other ops accept them
        state_idx = state_idx.float()
        
        card_indices_clamped = state_idx.clamp(0, self.num_cards - 1).long()
        one_hot_cards = F.one_hot(card_indices_clamped, num_classes=self.num_cards).float()
        flat_cards = one_hot_cards.view(one_hot_cards.size(0), -1)

        x = torch.cat([flat_cards, trump_onehot], dim=1)
        x = self.hidden(x)
        logits = self.policy_head(x)

        # ONNX export generally handles LogSoftmax correctly
        policy = F.log_softmax(logits, dim=1)
        return policy


# --- Conversion Function ---
def export_jass_dnn_to_onnx(model_path, output_path, num_cards, trump_dim, hidden_sizes, batch_size):
    """
    Loads a trained JassDNN PyTorch model from a file and converts it to ONNX format.

    Args:
        model_path (str): Path to the saved PyTorch model state dictionary (.pth file).
        output_path (str): Path to save the output ONNX model file (.onnx file).
        num_cards (int): Number of possible card values (must match the trained model).
        trump_dim (int): Dimension of trump one-hot encoding (must match the trained model).
        hidden_sizes (list): Hidden layer sizes (must match the trained model).
        batch_size (int): Batch size for the dummy input during conversion. Use 1 for a fixed batch size
                          or specify dynamic axes for variable batch size.
    """
    # --- Instantiate the model with the correct architecture ---
    model = JassDNN(num_cards=num_cards, trump_dim=trump_dim, hidden_sizes=hidden_sizes)

    # --- Load the trained model state dictionary ---
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

    # --- Create dummy input tensors ---
    dummy_state_idx = torch.randint(0, num_cards, (batch_size, 71), dtype=torch.long)
    dummy_trump_onehot = torch.eye(trump_dim).repeat(batch_size // trump_dim + 1, 1)[:batch_size]
    dummy_trump_onehot = dummy_trump_onehot[torch.randperm(batch_size)]

    # --- Define input and output names for the ONNX graph ---
    input_names = ["state_idx", "trump_onehot"]
    output_names = ["policy_log_probs"]

    # --- Define dynamic axes ---
    dynamic_axes = {
        "state_idx": {0: "batch_size"},
        "trump_onehot": {0: "batch_size"},
        "policy_log_probs": {0: "batch_size"}
    }

    # --- Export the model to ONNX format ---
    print("Exporting model to ONNX format...")
    try:
        torch.onnx.export(
            model,
            (dummy_state_idx, dummy_trump_onehot),
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


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a trained JassDNN PyTorch model to ONNX format.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved PyTorch model state dictionary file (.pth).')
    parser.add_argument('--output_path', type=str, default='jass_dnn_policy.onnx',
                        help='Path to save the output ONNX model file (.onnx).')
    parser.add_argument('--num_cards', type=int, default=71,
                        help='Number of possible card values used during training (must match trained model).')
    parser.add_argument('--trump_dim', type=int, default=7,
                        help='Dimension of trump one-hot encoding used during training (must match trained model).')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 256, 128],
                        help='Hidden layer sizes used during training (must match trained model). Provide as space-separated integers.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for the dummy input used during conversion. Set to 1 when using dynamic axes.')

    args = parser.parse_args()
    if args.hidden_sizes is None:
        args.hidden_sizes = []

    export_jass_dnn_to_onnx(
        args.model_path,
        args.output_path,
        num_cards=args.num_cards,
        trump_dim=args.trump_dim,
        hidden_sizes=args.hidden_sizes,
        batch_size=args.batch_size
    )