from utils.setup_helpers import PlantDataReader, SoilDataReader
from utils.constants import PlantResultDictKeys as PKeys
from utils.constants import PlantConstants

import pandas as pd
import math


class Plant:
    def __init__(self, cfg_dict):
        self.plant_reader = PlantDataReader(cfg_dict)
        self.soil_reader = SoilDataReader(cfg_dict, self.plant_reader)
        self.results = []
        self.total_interception_storage = 0  # [mm]  # is the storage term
        self.max_interception_capacity = self.plant_reader.total_leaf_area * PlantConstants.storage_capacity_leaves
        self.actual_interception_capacity = 0  # [mm]  # is the wind-related interception_capacity
        self.stem_capacity = self.max_interception_capacity*0.15
        self.wet_evaporation_factor = (1-math.e**(-0.75*self.plant_reader.horizontal_leaf_area_index))
        self.stem_storage = 0


    def simulate(self, e_pot, precipitation, wind_gusts, time):
        """
        This method executes one simulation step for the soil model
        Parameters
        ----------
        e_pot : float
            gross potential evaporation [mm/time_step] (reduction according to evaporation and transpiration will be
            calculated within this module)
        precipitation : float
            precipitation (reduced according to sheltering) [mm/timestep]
        wind_gusts: float
            wind speed to estimate interception [m/s]
        time: pd. Timestamp
            time in unix time stamp
        """
        self.append_result_buffer(time, e_pot)
        # first reduce by evaporation then calculate new interception, stemflow and throughfall
        self.calc_transpiration_demand_evaporation(e_pot)
        self.calc_interception(wind_gusts, precipitation)
        # self.calc_transpiration_demand_evaporation(e_pot)

    def calc_interception(self, wind_gusts, precipitation):
        self.actual_interception_capacity = (
                    self.max_interception_capacity * PlantConstants.wind_relation_coefficient ** wind_gusts)
        self.add_result(PKeys.actual_storage_capacity, self.actual_interception_capacity)

        calculated_throughfall = 0
        if self.total_interception_storage > self.actual_interception_capacity:
            # throughfall intensity [mm from all leaves] -> throughfall [mm]:
            # interception storage is in mm/(all leaves) ~ l on the total leaf area -> if 1 mm falls down -> 1l falls
            # on the ground below -> gets diverted on the whole surface.
            throughfall_rate = ((self.total_interception_storage-self.actual_interception_capacity) /
                                self.soil_reader.upper_soil_surface)
            calculated_throughfall += throughfall_rate
            self.total_interception_storage = self.actual_interception_capacity

        direct_throughfall = (self.plant_reader.direct_throughfall * precipitation -
                              PlantConstants.stemflow_fraction * precipitation)
        calculated_throughfall += direct_throughfall
        self.total_interception_storage += (1 - self.plant_reader.direct_throughfall) * precipitation
        self.stem_storage += PlantConstants.stemflow_fraction * (1 - self.plant_reader.direct_throughfall) * precipitation
        stemflow = 0
        if self.total_interception_storage > self.actual_interception_capacity:
            throughfall_rate = ((self.total_interception_storage - self.actual_interception_capacity) /
                                self.soil_reader.upper_soil_surface)
            calculated_throughfall += throughfall_rate
            self.total_interception_storage = self.actual_interception_capacity
        if self.stem_storage > self.stem_capacity:
            stemflow += self.stem_storage-self.stem_capacity
            self.stem_storage = self.stem_capacity
        self.add_result(PKeys.stemflow, stemflow/self.soil_reader.substrate_surface)
        self.add_result(PKeys.interception, self.total_interception_storage)
        self.add_result(PKeys.throughfall_rate, calculated_throughfall/self.soil_reader.upper_soil_surface)

    def calc_transpiration_demand_evaporation(self, e_pot):
        """
        calculates evaporation from interception and dependent on that, transpiration from dry leaves
        :param e_pot: potential evaporation [mm/t]
        """
        reduced_e_pot = e_pot  # [mm/t]

        reduced_e_pot -= self.evaporation_helper(reduced_e_pot)
        transpiration_demand = reduced_e_pot * self.plant_reader.resistive_transpiration_area  # [mm/t] * [m^2] = [l]

        self.add_result(PKeys.transpiration_demand, transpiration_demand)
        self.add_result(PKeys.reduced_evaporation, reduced_e_pot)

    def evaporation_helper(self, e_pot):
        #  TODO: this applies the evaporation as if the whole water would stand on a large field - no relation to actual
        #   vegetated area
        e_pot = e_pot*self.wet_evaporation_factor
        actual_evaporation = 0
        if e_pot <= self.total_interception_storage:  # not complete interception evaporates
            self.total_interception_storage -= e_pot  # [interception storage is in mm]
            actual_evaporation = e_pot  #
        elif e_pot > self.total_interception_storage:  # complete interception evaporates
            actual_evaporation = self.total_interception_storage
            self.total_interception_storage = 0
        self.add_result(PKeys.evap_interception, actual_evaporation, True)
        if self.stem_storage > 0:
            self.add_result(PKeys.evap_interception, min(self.stem_storage-e_pot, 0), True)
            self.stem_storage = min(self.stem_storage-e_pot, 0)
        return actual_evaporation

    def append_result_buffer(self, time, e_pot):
        self.results.append({PKeys.time: time, PKeys.throughfall_rate: 0,
                             PKeys.reduced_evaporation: e_pot,
                             PKeys.evap_interception: 0})

    def add_result(self, key, value, sum_up=False):
        if sum_up:
            self.results[-1][key] += value
        else:
            self.results[-1][key] = value

    def get_result(self, key):
        return self.results[-1][key]

    def export_results(self):
        res = pd.DataFrame(self.results)
        res[PKeys.time] = pd.to_datetime(res[PKeys.time], unit="s")
        res = res.add_prefix('V_')
        res = res.rename(columns={"V_" + PKeys.time: PKeys.time})
        res = res.set_index([PKeys.time])
        return res
