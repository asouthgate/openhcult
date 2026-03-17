from hcultutils.query import request_ctrl
import pandas as pd

def _submit_calibration_data(base_url, csv_file, version, created_at) -> int:

    dataframe = pd.read_csv(csv_file)

    data = {
        "swc": list(dataframe["swc"]),
        "sensor_val": list(dataframe["sensor_val"]),
        "swc_std": list(dataframe["swc_std"]),
        "version": version,
        "created_at": created_at
    }

    inserted_id = request_ctrl(
        "POST",
        f"{base_url}/calibration/response_curve_lookup",
        data,
    )
    assert inserted_id


def calibration_main(ctrl_url: str, action: str, args) -> int:
    base = ctrl_url.rstrip("/")
    if action == "submit_response_curve":
        return _submit_calibration_data(base, args.csv, args.version, args.created_at)