import math
from math import tanh
import numpy as np
from datetime import datetime

def s_curves1(t, x4):
    """
        Unit hydrograph ordinates for UH1 derived from S-curves.
    """

    if t <= 0:
        return 0
    elif t < x4:
        return (t/x4)**2.5
    else: # t >= x4
        return 1

def s_curves2(t, x4):
    """
        Unit hydrograph ordinates for UH2 derived from S-curves.
    """

    if t <= 0:
        return 0
    elif t < x4:
        return 0.5*(t/x4)**2.5
    elif t < 2*x4:
        return 1 - 0.5*(2 - t/x4)**2.5
    else: # t >= x4
        return 1

#function for pet using Hamon's method
def hamon_pet(date, temp, latitude):
    jdate = []
    for d in date:
        year, month, day = map(int, d.split('-'))
        try:
            formatted_date = datetime(year, month, day)
            jdate.append(formatted_date.timetuple().tm_yday)
        except ValueError:
            if month == 2 and day == 29:
                formatted_date = datetime(year, 2, 28)
                jdate.append(formatted_date.timetuple().tm_yday + 1)
            else:
                raise
    jdate = np.array(jdate)

    var_theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.0086 * (jdate - 186)))
    var_pi = np.arcsin(0.39795 * np.cos(var_theta))
    daylighthr = 24 - 24 / math.pi * np.arccos(
        (np.sin(0.8333 * math.pi / 180) + np.sin(latitude * math.pi / 180) * np.sin(var_pi)) /
        (np.cos(latitude * math.pi / 180) * np.cos(var_pi))
    )
    esat = 0.611 * np.exp(17.27 * temp / (237.3 + temp))
    potevap = 1 * 29.8 * daylighthr * (esat / (temp + 273.2)) # coeff_pet of 1 is used
    return potevap

def gr4j(precip, tavg, date, latitude, params, states = None, return_state = False):
    """
        Generated simulated streamflow for given rainfall and potential evaporation.

        :param precip: Catchment average rainfall.
        :type precip: array(float)
        :param potential_evap: Catchment average potential evapotranspiration.
        :type potential_evap: array(float)
        :param params: X parameters for the model.
        :type params: dictionary with keys X1, X2, X3, X4
        :param states: Optional initial state values.
        :type states: Dictionary with optional keys 'production_store', 'routing_store'.
        :param return_state: If true returns a dictionary containing 'production_store' and 'routing_store'. Default: False.
        :type return_state: boolean

        :return: Array of simulated streamflow.
    """
    
    # extract parameters
    if states is None:
        states = {}

    X1 = params['X1']
    X2 = params['X2']
    X3 = params['X3']
    X4 = params['X4']

    # calculate potential evapotranspiration
    potential_evap = hamon_pet(date, tavg, latitude)

    # calculate unit hydrograph ordinates
    nUH1 = int(math.ceil(X4))
    nUH2 = int(math.ceil(2.0*X4))

    uh1_ordinates = [0] * nUH1
    uh2_ordinates = [0] * nUH2

    UH1 = states.get('UH1', [0] * nUH1)
    UH2 = states.get('UH2', [0] * nUH2)

    for t in range(1, nUH1 + 1):
        uh1_ordinates[t - 1] = s_curves1(t, X4) - s_curves1(t-1, X4)

    for t in range(1, nUH2 + 1):
        uh2_ordinates[t - 1] = s_curves2(t, X4) - s_curves2(t-1, X4)

    production_store = states.get('production_store', 0) # S
    routing_store = states.get('routing_store', 0) # R

    qsim = []
    for P, E in zip(precip, potential_evap):

        if P > E:
            net_evap = 0
            scaled_net_precip = (P - E)/X1
            if scaled_net_precip > 13:
                scaled_net_precip = 13.
            tanh_scaled_net_precip = tanh(scaled_net_precip)
            reservoir_production = (X1 * (1 - (production_store/X1)**2) * tanh_scaled_net_precip) / (1 + production_store/X1 * tanh_scaled_net_precip)

            routing_pattern = P-E-reservoir_production
        else:
            scaled_net_evap = (E - P)/X1
            if scaled_net_evap > 13:
                scaled_net_evap = 13.
            tanh_scaled_net_evap = tanh(scaled_net_evap)

            ps_div_x1 = (2 - production_store/X1) * tanh_scaled_net_evap
            net_evap = production_store * (ps_div_x1) / \
                    (1 + (1 - production_store/X1) * tanh_scaled_net_evap)

            reservoir_production = 0
            routing_pattern = 0

        production_store = production_store - net_evap + reservoir_production

        percolation = production_store / (1 + (production_store/2.25/X1)**4)**0.25

        routing_pattern = routing_pattern + (production_store-percolation)
        production_store = percolation


        for i in range(0, len(UH1) - 1):
            UH1[i] = UH1[i+1] + uh1_ordinates[i]*routing_pattern
        UH1[-1] = uh1_ordinates[-1] * routing_pattern

        for j in range(0, len(UH2) - 1):
            UH2[j] = UH2[j+1] + uh2_ordinates[j]*routing_pattern
        UH2[-1] = uh2_ordinates[-1] * routing_pattern

        groundwater_exchange = X2 * (routing_store / X3)**3.5
        routing_store = max(0, routing_store + UH1[0] * 0.9 + groundwater_exchange)

        R2 = routing_store / (1 + (routing_store / X3)**4)**0.25
        QR = routing_store - R2
        routing_store = R2
        QD = max(0, UH2[0]*0.1+groundwater_exchange)
        Q = QR + QD

        qsim.append(Q)

    if return_state:
        return qsim, {
            'production_store': production_store,
            'routing_store': routing_store,
            'UH1': UH1,
            'UH2': UH2,
        }
    else:
        return qsim