import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import numpy as np
import torch

from nn_model import PINN


# ==========================================================
# Load model and scalers
# ==========================================================

def load_model():

    # ------------------------------------------------------
    # Load scalers
    # ------------------------------------------------------
    scalers = joblib.load("nn_scalers.pkl")

    # ------------------------------------------------------
    # Create model
    # ------------------------------------------------------
    model = PINN()

    model.load_state_dict(
        torch.load(
            "nn_model_weights.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model, scalers


# ==========================================================
# Prediction Function
# ==========================================================

def predict(
    model,
    scalers,
    T,
    N2,
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    MWC7,
    H2S,
    CO2,
):

    # ------------------------------------------------------
    # Get scalers
    # ------------------------------------------------------

    scaler_T = scalers["T"]
    scaler_N2 = scalers["N2"]
    scaler_C1 = scalers["C1"]
    scaler_C2 = scalers["C2"]
    scaler_C3 = scalers["C3"]
    scaler_C4 = scalers["C4"]
    scaler_C5 = scalers["C5"]
    scaler_C6 = scalers["C6"]
    scaler_C7 = scalers["C7"]
    scaler_MWC7 = scalers["MWC7"]
    scaler_H2S = scalers["H2S"]
    scaler_CO2 = scalers["CO2"]

    scaler_Pb = scalers["Pb"]

    # ------------------------------------------------------
    # Normalize inputs
    # ------------------------------------------------------

    T_norm = scaler_T.transform(np.array([[T]], dtype=np.float32))
    N2_norm = scaler_N2.transform(np.array([[N2]], dtype=np.float32))
    C1_norm = scaler_C1.transform(np.array([[C1]], dtype=np.float32))
    C2_norm = scaler_C2.transform(np.array([[C2]], dtype=np.float32))
    C3_norm = scaler_C3.transform(np.array([[C3]], dtype=np.float32))
    C4_norm = scaler_C4.transform(np.array([[C4]], dtype=np.float32))
    C5_norm = scaler_C5.transform(np.array([[C5]], dtype=np.float32))
    C6_norm = scaler_C6.transform(np.array([[C6]], dtype=np.float32))
    C7_norm = scaler_C7.transform(np.array([[C7]], dtype=np.float32))
    MWC7_norm = scaler_MWC7.transform(np.array([[MWC7]], dtype=np.float32))
    H2S_norm = scaler_H2S.transform(np.array([[H2S]], dtype=np.float32))
    CO2_norm = scaler_CO2.transform(np.array([[CO2]], dtype=np.float32))

    # ------------------------------------------------------
    # Create model input
    # ------------------------------------------------------

    x = np.hstack([
        T_norm,
        N2_norm,
        C1_norm,
        C2_norm,
        C3_norm,
        C4_norm,
        C5_norm,
        C6_norm,
        C7_norm,
        MWC7_norm,
        H2S_norm,
        CO2_norm
    ])

    x = torch.tensor(
        x,
        dtype=torch.float32
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    with torch.no_grad():

        y_norm = model(x)

    # ------------------------------------------------------
    # Convert back to original scale
    # ------------------------------------------------------

    y = scaler_Pb.inverse_transform(
        y_norm.detach().cpu().numpy()
    )

    return float(y[0, 0])
