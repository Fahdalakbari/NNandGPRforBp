import joblib
import numpy as np
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent


def load_model():
    # Load the trained GPR model and scalers
    model = joblib.load(PIPELINE_DIR / "gpr_model.pkl")
    scalers = joblib.load(PIPELINE_DIR / "scalers.pkl")

    return model, scalers


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
    CO2
):
    # Normalize every input using its training scaler
    input_values = {
        "T": T,
        "N2": N2,
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "C4": C4,
        "C5": C5,
        "C6": C6,
        "C7": C7,
        "MWC7": MWC7,
        "H2S": H2S,
        "CO2": CO2
    }

    normalized_inputs = []

    for feature, value in input_values.items():
        value_array = np.array([[value]], dtype=np.float64)
        normalized_value = scalers[feature].transform(value_array)
        normalized_inputs.append(normalized_value)

    # Preserve the same feature order used during GPR training
    model_input = np.hstack(normalized_inputs)

    # Generate the normalized GPR prediction
    prediction_normalized = model.predict(model_input)
    prediction_normalized = np.asarray(
        prediction_normalized
    ).reshape(-1, 1)

    # Convert normalized prediction back to psi
    prediction_psi = scalers["Pb"].inverse_transform(
        prediction_normalized
    )

    return float(prediction_psi[0, 0])
