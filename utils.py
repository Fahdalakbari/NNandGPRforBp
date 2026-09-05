import joblib
import numpy as np
import torch

from model import PINN


def load_models():
    """Load the NN, GPR, and the common preprocessing scalers."""
    scalers = joblib.load("scalers.pkl")

    nn_model = PINN()
    nn_model.load_state_dict(
        torch.load("nn_model.pth", map_location=torch.device("cpu"))
    )
    nn_model.eval()

    gpr_model = joblib.load("gpr_model.pkl")
    return nn_model, gpr_model, scalers


def predict(
    model_name, nn_model, gpr_model, scalers,
    T, N2, C1, C2, C3, C4, C5, C6, C7, MWC7, H2S, CO2
):
    """Predict bubble-point pressure in psi with the selected model."""
    feature_values = {
        "T": T, "N2": N2, "C1": C1, "C2": C2,
        "C3": C3, "C4": C4, "C5": C5, "C6": C6,
        "C7": C7, "MWC7": MWC7, "H2S": H2S, "CO2": CO2,
    }

    normalized = [
        scalers[name].transform(np.array([[value]], dtype=np.float64))
        for name, value in feature_values.items()
    ]
    x = np.hstack(normalized)

    if model_name == "Neural Network (NN)":
        with torch.no_grad():
            y_norm = nn_model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    elif model_name == "Gaussian Process Regression (GPR)":
        y_norm = np.asarray(gpr_model.predict(x)).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    y_psi = scalers["Pb"].inverse_transform(y_norm)
    return float(y_psi[0, 0])
