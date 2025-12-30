from src.functions import load_model, is_in_set, exp_value, features3, features5, features7, features12, predict
import pandas as pd
import joblib
import numpy as np


def main():
    #model, scaler = load_model("data/saved_models/test1.pk1")
    model, scaler = load_model("data/saved_models/gpr_m32-12.pk1")
    features = features12

    Z = int(input("Z: "))
    N = int(input("N: "))
    pred_be, pred_unc = predict(model, scaler, Z, N, features)

    print(f"Prediction: {pred_be:.3f} keV")
    print(f"Uncertainty: {pred_unc:.3f} keV")

    print(f"Range = {(pred_be - 2*pred_unc):.3f} keV -> {(pred_be + 2*pred_unc):.3f} keV")
    true_value = exp_value("data/processed/ame2020_parsed.csv", Z, N)
    print(f"Exp. value: {true_value} keV")
    try:
        print(f"Error: {(true_value - pred_be):.3f} keV")
        print(f"Error of Uncertainty: {(2*pred_unc - (true_value - pred_be)):.3f} keV")
        print("Is in training set:", is_in_set("data/processed/ame2020_train.csv", Z, N))

    except:
        None


if __name__ == "__main__":
    main()