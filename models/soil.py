from __future__ import annotations

import math
from abc import ABC
import logging
import pandas as pd

from utils.constants import SoilResultDictKeys as Keys
from utils.setup_helpers import weighted_average, SoilDataReader

RHO_W = 997  # kg/m^3
G = 9.981  # m/s^2
EPSILON = 0.0001  # vol%
gtime = 0  # just because I was to lazy to implement a working logging-filter
EXPONENTIAL_DECREASE = False  # otherwise linear decrease is applied when field capacity is reached
EXPONENTIAL_SKEW = 7  # 7 seems to be best


# TODO: avg_k might be an overestimation of possible in/outflow: avg_k = min(self.k, self.soil_below.k)


class Soil:

    # TODO: soil_type has to be index of the cgm dataframe
    def __init__(self, cfg_dict, time_step, plant_data_reader):
        self.plant_data_reader = plant_data_reader
        self.soil_reader = SoilDataReader(cfg_dict, self.plant_data_reader)
        self.upper_soil = UpperSoil(self.soil_reader, time_step)
        self.drainage = Drainage(self.soil_reader, time_step)
        self.substrate = Substrate(self.soil_reader, time_step)
        self.layers = [self.upper_soil, self.substrate, self.drainage]
        for layer_num in range(0, len(self.layers)):
            soil_above = None
            soil_below = None
            if layer_num > 0:
                soil_above = self.layers[layer_num - 1]
            if layer_num < len(self.layers) - 1:
                soil_below = self.layers[layer_num + 1]
            self.layers[layer_num].add_soil_above_soil_below(soil_above, soil_below)

    def simulate(self, plant_demand, e_pot, throughfall, additional_inflow, stemflow, time):
        """
        This method executes one simulation step for the soil model
        Parameters
        ----------
        plant_demand: float
            transpiration demand of all plants -> is taken from substrate layer
        e_pot : float
            gross potential evaporation [mm/time_step] (reduction according to two-layered vegetation will be
            calculated within this module)
        throughfall : float
            precipitation (reduced according to sheltering and interception) [mm/timestep]
        additional_inflow: float
            inflow which runs into drainage_layer [m^3/time_step]
        stemflow: float
            stemflow into the substrate (will not be infiltrated but reaches the substrate directly
        time: Timestamp
            time in unix time stamp
        """

        global gtime
        gtime = time.strftime("%d %H:%M:%S")
        additional_inflow /= 1000  # liters -> m^3
        plant_demand /= 1000  # mm->m
        e_pot /= 1000  # mm->m
        throughfall /= 1000  # mm -> m
        stemflow /= 1000  # mm -> m

        for layer in self.layers:
            layer.append_result_buffer(time)
        self.upper_soil.apply_rain(throughfall)
        #print(self.drainage.catch_basin_volume)
        self.drainage.apply_additional_inflow(additional_inflow)  # TODO: MAYBE change it to the end of procedure
        for layer in self.layers:
            layer.recalculate_soil_characteristics()
            logging.info(layer)

        for layer in self.layers:
            layer.calc_infiltration()
        for layer in self.layers:
            layer.calc_capillary_rise()
        # print("calc transpiration")
        self.substrate.calc_transpiration(plant_demand)
        self.upper_soil.calc_direct_evaporation(e_pot)
        self.drainage.calc_inner_infiltration()  # was formerly in apply_calculations
        accepted_flow = None
        for layer in reversed(self.layers):  # reversed because drainage is normally the biggest source
            layer.apply_calculations()
            # print(layer)
        self.substrate.apply_stemflow(stemflow)
        # print("END:")
        # for layer in self.layers:
        # print(layer)

    def export_results(self):
        res = [i.export_results() for i in self.layers]
        r_n = res[0]
        for r in res[1:]:
            r = r.reset_index()
            r_n = pd.merge(r_n, r, on=Keys.time)
        r_n = r_n.set_index(Keys.time)
        return r_n
    # TODO: implement a conti-checker function (infiltration + evaporation - rain = 0) IDEA: can be a function of the
    #  value -> large values receive more water


class SoilBase(ABC):
    def __init__(self, cfg_dict, soil_data_reader: SoilDataReader, time_step):

        self.soil_below = None
        self.soil_above = None
        self.results = []
        self.time_step = time_step
        self.cfg = cfg_dict
        self.density = cfg_dict["bulk_density"]
        self.soil_type = cfg_dict["soil_type"]
        self.soil_thickness = cfg_dict["thickness"]
        self.surface = cfg_dict["surface"]
        self.volume = calc_volume(self.surface, self.soil_thickness)
        self.s = cfg_dict["initial_S"]
        self.vgm_params = soil_data_reader.van_genuchten
        self.k_0 = self.vgm_params.loc[self.soil_type, "K_0"]  # [cm/d]
        self.theta_r = self.vgm_params.loc[self.soil_type, "theta_r"]
        self.theta_s = self.vgm_params.loc[self.soil_type, "theta_s"]
        self.n = self.vgm_params.loc[self.soil_type, "n"]
        self.m = 1 - 1 / self.n
        self.x = self.vgm_params.loc[self.soil_type, "x"]
        self.theta = 0
        self.set_theta("S", self.s)
        self.alpha = self.vgm_params.loc[self.soil_type, "alpha"]  # [1/hPa]
        self.psi_m = 0
        self.k = 0
        self.bulk_density = cfg_dict["bulk_density"]
        self.field_capacity = soil_data_reader.get_field_capacity(self.soil_type, self.bulk_density)
        self.theta_d = calc_theta_from_psi(4 * 10 ^ 3, self.theta_r, self.theta_s, self.alpha, self.n, self.m)
        self.k_s = soil_data_reader.get_k_s(self.soil_type, self.bulk_density)
        self.pwp = calc_theta_from_psi(10 ** 4.2, self.theta_r, self.theta_s, self.alpha, self.n,
                                       self.m)  # pwp in vol %
        self.slope_transpiration_curve_d = 1 / (self.theta_d - self.pwp)
        self.intercept_transpiration_curve_d = - self.slope_transpiration_curve_d * self.pwp  # b
        self.slope_transpiration_curve = 1 / (self.field_capacity - self.pwp)  # a  LINEAR: f=a*t+b
        self.intercept_transpiration_curve = - self.slope_transpiration_curve * self.pwp  # b
        self.factor_assertion = - 1 / (math.e ** (EXPONENTIAL_SKEW * self.pwp) - math.e ** (
                EXPONENTIAL_SKEW * self.field_capacity))  # a*e^(7*t)+b
        self.addend_assertion = -self.factor_assertion * math.e ** (EXPONENTIAL_SKEW * self.field_capacity)
        self.theta_max = soil_data_reader.get_total_pore_volume(self.soil_type)
        self.available_pore_volume = 0
        self.available_water_volume = 0
        self.recalculate_soil_characteristics(False)

    def __str__(self):
        return f"{self.__class__.__name__}: S={self.s}, theta={self.theta}, psi_m={self.psi_m}, k={self.k}"

    def add_soil_above_soil_below(self, soil_above: SoilBase, soil_below: SoilBase):
        self.soil_above = soil_above
        self.soil_below = soil_below

    def calc_infiltration(self):
        assertion = reduced_assertion(self.soil_above)

        dz = (self.soil_thickness + self.soil_above.soil_thickness) / 2
        dh = self.psi_m - self.soil_above.psi_m + hydrostatic_pressure(dz)
        avg_k = weighted_average([self.k, self.soil_above.k], [self.soil_thickness, self.soil_above.soil_thickness])
        infiltration = darcy(dh, dz, avg_k) * self.time_step

        if infiltration < 0:
            infiltration = 0
        logging.debug(f"{gtime} INFILTRATION {self.__class__.__name__}: rate = {infiltration:.4e}, "
                      f"theta:{self.soil_above.theta:.4f}, fc={self.soil_above.field_capacity:.4f},"
                      f" ass = {assertion:4f}")
        self.add_result(Keys.calc_inf, infiltration * assertion)

    def calc_capillary_rise(self):
        assertion = reduced_assertion(self.soil_below)

        dz = (self.soil_thickness + self.soil_below.soil_thickness) / 2
        dh = self.psi_m - self.soil_below.psi_m - hydrostatic_pressure(dz)
        avg_k = weighted_average([self.k, self.soil_below.k], [self.soil_thickness, self.soil_below.soil_thickness])

        capillary_rise = darcy(dh, dz, avg_k)
        capillary_rise *= self.time_step

        if capillary_rise < 0:
            capillary_rise = 0
        logging.debug(f"{gtime} CAPILLARY {self.__class__.__name__}: rate = {capillary_rise:.4e}, "
                      f"theta:{self.soil_below.theta:.4f}, fc={self.soil_below.field_capacity:.4f}, "
                      f"ass = {assertion:4f}")
        self.add_result(Keys.calc_cap, capillary_rise * assertion)

    def recalculate_soil_characteristics(self, add_to_results=True):

        #  % of maximum saturation
        self.s = calc_s(self.theta, self.theta_r, self.theta_s)
        self.k = min(calc_k(self.k_0, self.s, self.x, self.m), self.k_s)  # gives m/s
        # hPa -> Pa (just because SI)
        self.psi_m = calc_psi_m(theta=self.theta, theta_r=self.theta_r, theta_s=self.theta_s, m=self.m, n=self.n,
                                alpha=self.alpha)
        self.available_pore_volume = (self.theta_max - self.theta) * self.volume
        self.available_water_volume = (self.theta - self.pwp) * self.volume
        # TODO: might be wrong -> could also be theta_r
        if add_to_results:
            self.add_result(Keys.psi_m, self.psi_m)
            self.add_result(Keys.calc_k, self.k)
            self.add_result(Keys.theta, self.theta)
            self.add_result(Keys.S, self.s)

    def append_result_buffer(self, time):
        self.results.append({Keys.time: time, Keys.calc_cap: 0})

    def add_result(self, key, value, sum_up=False):
        if sum_up:
            self.results[-1][key] += value
        else:
            self.results[-1][key] = value

    def get_result(self, key):
        return self.results[-1][key]

    def set_theta(self, input_type, value):
        """
        calculates a theta or takes the given theta as input.

        Parameters
        ----------
        input_type: str
            Either "theta" or "S"
        value: float
            Value for S or theta
        """
        if input_type == "theta":
            self.theta = value
        elif input_type == "S":
            self.theta = calc_theta_from_s(value, self.theta_r, self.theta_s)
            self.s = value

    def apply_calculations(self):
        """
        This method applies the results and checks for the fulfillment of the continuity equation
        :return upwards actual flow
        """
        inflow = self.get_result(Keys.calc_inf)
        inflow += self.get_result(Keys.calc_cap)

        if self.soil_below is None:  # Drainage
            """
            Procedure: first calculate new theta according to overall flow then calculate inner change of water (from
            saturated lower drainage layer up to the unsaturated layer
            """
            outflow = self.soil_above.get_result(Keys.calc_cap)

            resulting_flow = inflow - outflow
            logging.info(
                f"{gtime} {self.__class__.__name__}: outflow = {outflow}, inflow = {inflow} => {resulting_flow}")
            real_flow = self.update_theta(resulting_flow)
            self.add_result(Keys.outflow, outflow)
            self.add_result(Keys.inflow, inflow)
            self.add_result(Keys.resulting_flow, real_flow)
            inner_inflow = self.get_result(Keys.calc_inner_inf)  # from unsaturated to saturated soil
            inner_outflow = self.get_result(Keys.calc_inner_cap)  # from saturated to unsaturated soil
            resulting_inner_flow = inner_inflow - inner_outflow
            self.update_saturated_zone(resulting_inner_flow)
        elif self.soil_above is None:  # upper soil
            outflow = self.get_result(Keys.calc_evap)
            outflow += self.soil_below.get_result(Keys.calc_inf)
            resulting_flow = inflow - outflow
            real_flow = self.update_theta(resulting_flow)
            self.add_result(Keys.outflow, outflow)
            self.add_result(Keys.inflow, inflow)
            self.add_result(Keys.resulting_flow, real_flow)
            frac_flow = 1
            if resulting_flow != 0:
                frac_flow = real_flow / resulting_flow
            self.liquid_level -= frac_flow * self.get_result(Keys.calc_inf)
            logging.info(
                f"{gtime} {self.__class__.__name__}: outflow = {outflow}, inflow = {inflow} => {resulting_flow} "
                f"FRAC_FLOW = {frac_flow}, liquid_level = {self.liquid_level}")
            self.add_result(Keys.liquid_level, self.liquid_level)
        else:  # substrate or other fully-surrounded layers
            outflow = self.soil_above.get_result(Keys.calc_cap)
            outflow += self.soil_below.get_result(Keys.calc_inf)
            outflow += self.get_result(Keys.calc_trans)
            resulting_flow = inflow - outflow
            real_flow = self.update_theta(resulting_flow)
            self.add_result(Keys.outflow, outflow)
            self.add_result(Keys.inflow, inflow)
            self.add_result(Keys.resulting_flow, real_flow)
            frac_flow = 1
            if resulting_flow != 0:
                frac_flow = real_flow / resulting_flow
            logging.info(
                f"{gtime} {self.__class__.__name__}: outflow = {outflow}, inflow = {inflow} => {resulting_flow} "
                f"FRAC_FLOW = {frac_flow}")
            self.add_result(Keys.applied_trans, self.get_result(Keys.calc_trans) * frac_flow * self.surface * 1000)

    def update_theta(self, flow):
        """
        updates theta and returns an actual flow (actual flow is dependent on current theta -> pore volume is limited)
        :param flow: flow rate [mm/t]
        :return: actual flow: [mm/t]
        """
        flux = flow_rate_to_flux(flow, self.surface)
        if flow > 0:  # water flows in: soil theta changes
            d_theta = self.theta_max - self.theta
            available_pore_volume = d_theta * self.volume
            if flux > available_pore_volume:
                logging.debug(
                    f"{gtime} flux ({flux}) > available pore volume ({available_pore_volume} -> updating flux to"
                    f" available pore-volume")
                flux = min(available_pore_volume, flux)
        elif flow < 0:  # water flows out: soil_theta reduces
            d_theta = self.theta - self.pwp
            available_water_volume = d_theta * self.volume
            if abs(flux) > available_water_volume:
                logging.debug(
                    f" {gtime} flux ({flux}) > available water volume ({available_water_volume} -> updating flux to"
                    f" available water_volume")
                flux = max(-available_water_volume, flux)
        if flux != 0:
            self.theta += flux / self.volume
        return flux / self.surface  # TODO: this return value might be used for conty-checker!

    def calc_inner_infiltration(self):
        pass

    def update_saturated_zone(self, flow):
        pass

    def export_results(self):
        res = pd.DataFrame(self.results)
        res = res.add_prefix(self.__class__.__name__[0:1] + "_")
        res = res.rename(columns={self.__class__.__name__[0:1] + "_" + Keys.time: Keys.time})
        res.set_index([Keys.time])
        return res


class UpperSoil(SoilBase):
    def __init__(self, soil_reader: SoilDataReader, time_step):
        super().__init__(soil_reader.upper_soil, soil_reader, time_step)
        self.free_area = soil_reader.free_soil_surface
        self.surface_resistance = self.cfg["surface_resistance"]
        self.liquid_level = 0
        self.max_liquid_level = self.cfg["max_standing_water"]

    def apply_rain(self, throughfall):
        """
        adds the rain to the upper soil -> will be reduced in the same calculation step by infiltration => it has to be
            called before calc_infiltration
        :param precipitation: precipitation rate [mm/t] is the throughfall + direct_throughfall
        :return: None
        """
        self.liquid_level += throughfall
        overflow = 0
        if self.liquid_level >= self.max_liquid_level:
            overflow = (self.max_liquid_level - self.liquid_level) * self.surface
            self.liquid_level = self.max_liquid_level
        self.add_result(Keys.liquid_level, self.liquid_level)
        self.add_result(Keys.overflow, overflow)

    def calc_infiltration(self):
        infiltration = 0
        if self.liquid_level > 0:  # infiltration from standing water -> upper soil
            dz = self.soil_thickness / 2
            dh = self.psi_m + hydrostatic_pressure(self.liquid_level + dz)
            infiltration = darcy(dh, dz, self.k) * self.time_step
            infiltration = min(infiltration, self.liquid_level / self.time_step)
            # otherwise more water could infiltrate than there
            # is on surface
            logging.info(f"{gtime}: UPPER SOIL INFILTRATION: dh={dh}, dz={dz}, k = {self.k}infiltration={infiltration} "
                         f"liquid_level = {self.liquid_level}")

        self.add_result(Keys.calc_inf, infiltration)

    def calc_direct_evaporation(self, e_pot):
        evaporation = 0
        reduced_e_pot = e_pot
        if self.liquid_level > 0:
            evaporation = min(self.liquid_level, reduced_e_pot)
            self.liquid_level -= evaporation
            self.add_result(Keys.liquid_level,
                            self.liquid_level)  # evaporation from liquid surface is not be restricted
            reduced_e_pot -= evaporation
            evaporation = 0  # this is important, otherwise, the loss in water is applied to the upper soil!
        assertion = reduced_assertion(self, False)
        evaporation += reduced_e_pot * self.surface_resistance * assertion * self.free_area / self.surface
        logging.debug(
            f"{gtime} DIRECT EVAPORATION: e= ass({assertion}) * surface_res({self.surface_resistance})* "
            f"e_pot({reduced_e_pot})")

        self.add_result(Keys.calc_evap, evaporation)


class Substrate(SoilBase):
    def __init__(self, soil_reader: SoilDataReader, time_step):
        super().__init__(soil_reader.substrate, soil_reader, time_step)

    def calc_transpiration(self, plant_demand):
        assertion = reduced_assertion(self, False)
        self.add_result(Keys.calc_trans, assertion * plant_demand / self.surface)
        if assertion == 0:
            logging.info("Plant does not receive any water!")
            self.add_result(Keys.no_plant_water, 1)

    def apply_stemflow(self, stemflow):
        """
        This method should be called at the end of each simulation step. Otherwise, the stemflow of the same
        simulation step reaches the substrate immediately
        :param stemflow: float [m^3/t]
        :return: None
        """
        absorbed_sf = self.update_theta(stemflow)
        self.add_result(Keys.stemflow_absorbed, absorbed_sf)

    def append_result_buffer(self, time):
        super().append_result_buffer(time)
        self.results[-1][Keys.no_plant_water] = 0


class Drainage(SoilBase):
    def __init__(self, soil_reader: SoilDataReader, time_step):
        super().__init__(soil_reader.drainage, soil_reader, time_step)
        self.overall_thickness = self.soil_thickness  # this does not change over time
        self.overflow_height = self.cfg["outflow_height"]
        self.saturated_soil_thickness = 0
        self.catch_basin_surface = self.cfg["catch_basin_surface"]  # left and right catch basin
        self.catch_basin_volume = 0  # volume of water
        self.catch_basin_volume_max = self.catch_basin_surface * self.overflow_height
        self.total_area = self.catch_basin_surface + self.surface
        self.set_saturated_layer(self.cfg["initial_water_level"])  # thickness is water level

    def set_saturated_layer(self, liquid_level):
        self.saturated_soil_thickness = min(liquid_level, self.overflow_height)
        if liquid_level > self.overflow_height:
            logging.info(
                "initial liquid level > overflow height -> Setting saturated soil thickness to overflow height")
        self.soil_thickness = self.overall_thickness - self.saturated_soil_thickness
        self.volume = calc_volume(self.soil_thickness, self.surface)
        self.catch_basin_volume = self.catch_basin_surface * liquid_level

    def apply_additional_inflow(self, flux):
        """
        adds the inflow to the saturated layer. If the resulting height is higher, overflow occurs immediately.
        :param flux: [m^3/t]
        :return: None
        """
        # inflow is taken up immediately, because no hydrodynamic model which could calculate a higher inflow pressure
        # was implemented. capillary rise is calculated afterward.
        d_theta = self.theta_max - self.theta
        oversupply_of_water = 0
        accepted_flux = flux
        d_theta_star = self.calc_d_theta_star(d_theta)
        #available_pore_volume = d_theta * volume_to_overflow_height
        available_pore_volume = d_theta_star*self.total_area*(self.overflow_height - self.saturated_soil_thickness)
        #available_pore_volume += self.catch_basin_volume_max - self.catch_basin_volume
        if available_pore_volume < flux:
            oversupply_of_water = flux - available_pore_volume
            accepted_flux = available_pore_volume

        new_water_level = self.saturated_soil_thickness + (accepted_flux/self.total_area)/d_theta_star
        self.soil_thickness = self.overall_thickness - new_water_level
        self.catch_basin_volume = new_water_level*self.catch_basin_surface
        #self.volume -= accepted_flux / d_theta
        self.volume = self.soil_thickness * self.surface
        self.volume = max(self.volume, (self.overall_thickness - self.overflow_height) * self.surface)  # for rounding errors
        logging.debug(
            f"{gtime} ADDITIONAL INFLOW: f:{flux:.4e} -> ({accepted_flux:.3e}) av.PV:{available_pore_volume:3e}, "
            f"d_theta:{d_theta:.4f}, overflow: {oversupply_of_water:.3e} soil_thickness: {self.soil_thickness}->{self.volume/self.surface:.4f} "
            f"{(self.soil_thickness - (self.overall_thickness - self.overflow_height))}"
            f"d_theta* = {d_theta_star:.4f}, catchment vol = {self.catch_basin_volume:.4f}")

        self.saturated_soil_thickness = self.overall_thickness - self.soil_thickness
        self.add_result(Keys.overflow, oversupply_of_water, True)
        self.add_result(Keys.accepted_flux, accepted_flux)
        self.add_result(Keys.available_pore_volume, available_pore_volume)

    def calc_capillary_rise(self):
        """
            calculates capillary rise between saturated and unsaturated soil of drainage
        :return:
        """
        if self.theta >= self.theta_max:
            logging.debug(f"{gtime} unsaturated soil has the same theta as saturated soil -> NO CAPILLARY RISE")
            self.add_result(Keys.calc_inner_cap, 0)
            return

        dz = self.soil_thickness / 2
        dh = self.psi_m - hydrostatic_pressure(dz)
        capillary_rise = darcy(d_h=dh, d_z=dz, k=self.k)
        capillary_rise *= self.time_step

        assertion = reduced_assertion(self)
        if capillary_rise < 0:
            # logging.debug(f"{gtime} capillary_rise in {self.__class__.__name__} is < 0 -> could be infiltration")
            capillary_rise = 0
        self.add_result(Keys.calc_inner_cap, capillary_rise * assertion)
        logging.info(f"{gtime} inner cap = {capillary_rise}, dh={dh},dz= {dz}, k={self.k} ass = {assertion},"
                     f" theta = {self.theta}, field_capacity={self.field_capacity}")

    def calc_inner_infiltration(self):
        assertion = reduced_assertion(self)
        dz = self.soil_thickness/2
        dh = hydrostatic_pressure(dz) - self.psi_m
        infiltration = darcy(d_h=dh, d_z=dz, k=self.k)
        infiltration *= self.time_step
        if infiltration < 0:
            infiltration = 0
        self.add_result(Keys.calc_inner_inf, infiltration * assertion)

        # logging.info(f"{gtime} inner infiltration = {infiltration}, dh={dh},dz= {dz}, k={self.k}")

    def calc_d_theta_star(self, d_theta):
        volume_to_overflow_height = calc_volume(self.surface,
                                                (self.soil_thickness - (self.overall_thickness - self.overflow_height)))
        total_volume_to_of_height = volume_to_overflow_height + (self.catch_basin_volume_max - self.catch_basin_volume)
        available_pore_volume = d_theta * volume_to_overflow_height
        if volume_to_overflow_height == 0 or total_volume_to_of_height == 0:
            d_theta_star = 1
        else:
            d_theta_star = ((available_pore_volume + (self.catch_basin_volume_max - self.catch_basin_volume)) /
                            (volume_to_overflow_height + total_volume_to_of_height))
        #print(f"d_theta* = {d_theta_star}, d_theta = {d_theta}, vol_to_overflow_height = {volume_to_overflow_height}, "
        #      f"totOF = {total_volume_to_of_height}, cbvm = {self.catch_basin_volume_max}, cbv = {self.catch_basin_volume}")
        return d_theta_star

    def update_saturated_zone(self, flow):
        """
        updates the thickness of the saturated and unsaturated layers of soil. Theta only changes in the
        unsaturated soil. The water content (s) of saturated soil is 100%, while for the unsaturated soil it changes.
        The saturated layer is drained until moisture content of theta_unsaturated is reached.
        Gives a new thickness of unsaturated soil, where the additional water is distributed evenly.
        Parameters
        ----------
        flow : float [mm/t] is interpreted as a volume since it was already multiplied with time-step
            positive = infiltration
            negative = capillary rise
        """

        d_theta = self.theta_max - self.theta
        flux = flow_rate_to_flux(flow, self.surface)
        d_theta_star = self.calc_d_theta_star(d_theta)
        if flow > 0:  # infiltration -> water comes from unsaturated soil. Volume and theta changes
            available_water_volume = (self.theta - self.pwp) * calc_volume(self.surface, self.soil_thickness)
            #available_pore_volume = d_theta_star * calc_volume(self.surface, self.soil_thickness)
            available_pore_volume = d_theta_star * self.total_area*(self.overflow_height-self.saturated_soil_thickness)
            if available_pore_volume < flux or available_water_volume < flux:
                self.add_result(Keys.overflow, max(flux - available_water_volume, flux - available_pore_volume), True)
                flux = min(available_pore_volume, available_water_volume, flux)
            flow = flux / self.total_area
            self.saturated_soil_thickness += flow / d_theta_star

            if self.saturated_soil_thickness > self.overflow_height:
                self.add_result(Keys.overflow, d_theta * calc_volume(self.surface, (  # This never happens!
                        self.saturated_soil_thickness - self.overflow_height)), True)
                self.saturated_soil_thickness = self.overflow_height
            self.soil_thickness = self.overall_thickness - self.saturated_soil_thickness
            # available_water_volume = (self.theta - self.pwp) * calc_volume(self.surface, self.soil_thickness)
            self.theta -= flux / self.volume  # (available_water_volume-flux)/self.volume
            self.volume = self.surface * self.soil_thickness


        elif flow < 0:  # capillary rise-> water source is the saturated soil
            available_water_volume = (self.theta_max - self.pwp) * calc_volume(self.surface,
                                                                               self.saturated_soil_thickness)
            available_water_volume += self.catch_basin_volume
            available_pore_volume = d_theta * calc_volume(self.surface, self.soil_thickness)
            if flux < -available_water_volume or flux < -available_pore_volume:
                flux = max(-available_water_volume, flux)
            # the volume, which changes from saturated to unsaturated given the d_theta
            downward_flow = flux/self.total_area
            # self.saturated_soil_thickness += flow / d_theta  # flux is negative -> sat. soil thickness decreases!
            self.saturated_soil_thickness += downward_flow / d_theta_star  # flux < 0 -> sat. soil thickness decreases!
            self.soil_thickness = self.overall_thickness - self.saturated_soil_thickness
            self.theta -= flux/(self.soil_thickness*self.surface)

        self.catch_basin_volume = self.catch_basin_surface * self.saturated_soil_thickness
        self.volume = calc_volume(self.surface, self.soil_thickness)
        self.add_result(Keys.u_soil_thickness, self.soil_thickness)
        self.add_result(Keys.s_soil_thickness, self.saturated_soil_thickness)

    def append_result_buffer(self, time):
        super().append_result_buffer(time)
        self.results[-1][Keys.overflow] = 0


def reduced_assertion(soil: SoilBase, percolation=True):  # For transpiration and evaporation theta_d is used
    if percolation:
        if EXPONENTIAL_DECREASE:
            return reduced_assertion_helper(soil.theta, soil.pwp, soil.field_capacity,
                                            soil.factor_assertion, soil.addend_assertion)
        else:
            return reduced_assertion_helper(soil.theta, soil.pwp, soil.field_capacity,
                                            soil.slope_transpiration_curve, soil.intercept_transpiration_curve_d)
    else:
        if EXPONENTIAL_DECREASE:
            return reduced_assertion_helper(soil.theta, soil.pwp, soil.field_capacity,
                                            soil.factor_assertion, soil.addend_assertion)
        else:
            return reduced_assertion_helper(soil.theta, soil.pwp, soil.field_capacity,
                                            soil.slope_transpiration_curve_d, soil.intercept_transpiration_curve_d)


def reduced_assertion_helper(theta, pwp, threshold, slope, interception):
    if theta > threshold:
        assertion = 1
    elif pwp < theta < threshold:
        if EXPONENTIAL_DECREASE:
            assertion = slope * math.e ** (theta * EXPONENTIAL_SKEW) + interception
        else:
            assertion = (slope * theta + interception)
    else:
        assertion = 0
    return assertion


def darcy(d_h, d_z, k):
    return k * d_h / (d_z * (RHO_W * G))


def hydrostatic_pressure(h):  # h in metres gives Pa = N/m^2
    return h * RHO_W * G


def calc_k(k_0, s, x, m):
    # cm/d-> m/s
    k = (k_0 * s ** x * (1 - (1 - s ** (1 / m)) ** m) ** 2) / (100 * 3600 * 24)
    if pd.isna(k):
        raise ValueError("ERROR")
    return k


def calc_s(theta, theta_r, theta_s):
    return (theta - theta_r) / (theta_s - theta_r)


def calc_psi_m(theta, theta_r, theta_s, m, n, alpha):
    return ((((theta_r - theta_s) / (theta_r - theta)) ** (1 / m) - 1) ** (1 / n)) / alpha * 100


def calc_theta_from_s(s, theta_r, theta_s):
    return theta_r * (-s) + theta_r + s * theta_s


def calc_volume(area, depth):
    return area * depth


def calc_theta_from_psi(psi_m, theta_r, theta_s, alpha, n, m):
    return theta_r + (theta_s - theta_r) / ((1 + (alpha * psi_m) ** n) ** m)


def flow_rate_to_flux(flow_rate, area):
    return flow_rate * area
