from hcultutils.fetch_data import fetch_data
from hcultinf.training import filter_confirmed_watering_events


def inference_train_main(ctrl_url: str, args) -> int:
    plant_name = getattr(args, "plant_name", None)
    series, observations = fetch_data(
        ctrl_url, args.start_utc, args.end_utc, plant_name=plant_name, limit=args.limit
    )
    if not series or not observations:
        print("No data found.")
        return 0

    confirmed = filter_confirmed_watering_events(observations)
    print(f"Found {len(confirmed)} confirmed watering events.")

    raise NotImplementedError("Parameter optimisation not yet implemented.")
