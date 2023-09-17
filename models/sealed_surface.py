from utils.constants import RoofResultDictKeys as Keys
from utils.setup_helpers import weighted_average
import pandas as pd


class ConnectedImpermeableArea:
    def __init__(self, cfg):
        self.areas = [i["area"] for i in cfg["impermeable_connected_area"].values()]
        self.runoff_coefficients = [i["runoff_coefficient"] for i in cfg["impermeable_connected_area"].values()]
        self.runoff_area = weighted_average(self.areas, self.runoff_coefficients) * sum(self.runoff_coefficients)
        reciprocal_r_coefficients = [1 - i for i in self.runoff_coefficients]
        self.evaporation_area = weighted_average(self.areas, reciprocal_r_coefficients) * sum(reciprocal_r_coefficients)
        self.results = []

    def append_results_buffer(self, time):
        self.results.append({Keys.time: time})

    def add_results(self, key, value):
        self.results[-1][key] = value

    def get_runoff(self):
        return self.results[-1][Keys.runoff]

    def calc_runoff(self, time, gross_precipitation):
        """
        applies DWA A-138
        Parameters
        ----------
        gross_precipitation: float
            gross_precipitation in [mm/t]
        time: in unix, just for the results
        :return float runoff [l/t]
        """
        self.append_results_buffer(time)
        self.add_results(Keys.runoff, gross_precipitation * self.runoff_area)
        self.add_results(Keys.evaporation, gross_precipitation * self.evaporation_area)

    def export_results(self):
        res = pd.DataFrame(self.results)
        res[Keys.time] = pd.to_datetime(res[Keys.time], unit="s")
        res = res.add_prefix('CA_')
        res = res.rename(columns={"CA_" + Keys.time: Keys.time})
        res = res.set_index([Keys.time])
        return res
