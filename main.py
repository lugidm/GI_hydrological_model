import sys
import json
import logging
from models.plant import Plant
from models.soil import Soil
from models.sealed_surface import ConnectedImpermeableArea
from utils.setup_helpers import WeatherDataReader
from utils.constants import PlantResultDictKeys as PKeys, SoilResultDictKeys as SKeys
import pandas as pd
import pytz
import time as ttt
TIME_STEP = 60  # sec
RAINFALL_TIMESTEP = 3600  # in seconds
cfg = {}
TOTAL_SIM_TIME = (30+31+31+8)*24*3600
START_TIME = 1685577600     # 2023-06-01 00:00:00+00:00


def main(argv):
    global cfg
    with open(argv[1], mode='r') as file:
        cfg = json.load(file)
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(message)s',
    #                    datefmt="%d %H:%M:%S", filename="log.log", filemode="w")

    plant_model = Plant(cfg)
    soil_model = Soil(cfg, TIME_STEP, plant_model.plant_reader)
    connected_area = ConnectedImpermeableArea(cfg)
    weather_data = WeatherDataReader(cfg, TIME_STEP)
    time = START_TIME

    start = ttt.time()
    while time < START_TIME + TOTAL_SIM_TIME:
        timestamp = pd.to_datetime(time, unit='s')
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
        precipitation = weather_data.get_precipitation(timestamp)
        e_pot = weather_data.get_e_pot(timestamp)
        connected_area.calc_runoff(time=timestamp, gross_precipitation=precipitation)
        plant_model.simulate(e_pot=e_pot, precipitation=precipitation * cfg["rain_exposure"],
                             wind_gusts=weather_data.get_wind_gusts(timestamp), time=timestamp)
        soil_model.simulate(plant_demand=plant_model.get_result(PKeys.transpiration_demand),
                            e_pot=e_pot, throughfall=plant_model.get_result(PKeys.throughfall_rate),
                            additional_inflow=connected_area.get_runoff(),
                            stemflow=(plant_model.get_result(PKeys.stemflow)),
                            time=timestamp)

        logging.info(f"------------ simulation_step {time - START_TIME}, precipitation={precipitation}, e_pot={e_pot} "
                     f"------------")
        time += TIME_STEP
    end = ttt.time()
    results = [weather_data.export_results(), connected_area.export_results(), plant_model.export_results(), soil_model.export_results()]
    r_n = results[0]
    for r in results[1:]:
        r = r.reset_index()
        r_n = pd.merge(r_n, r, on=PKeys.time)
    r_n = r_n.set_index(PKeys.time)
    print(r_n)
    print(f"Duration: {(end-start)}")
    r_n.to_csv("results.csv")


if __name__ == "__main__":
    main(sys.argv)
