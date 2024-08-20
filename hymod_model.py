'''
HYMOD model 
Reference:
- (Boyle, 2001) https://repository.arizona.edu/bitstream/handle/10150/290657/azu_td_3023521_sip1_m.pdf?sequence=1&isAllowed=y
- (Roy et al, 2017) https://hess.copernicus.org/articles/21/879/2017/hess-21-879-2017.pdf
- Matlab code by Sungwook Wi
'''
#import libraries
import numpy as np
import pandas as pd
import math
# hymod function
def hymod(pars, p, temp, date, latitude, routing):
    '''
    hymod function Input:
    - pars: parameter vector
    - p & temp: precipitation & temperature time series
    - date: date in YYYY-MM-DD
    - latitude: centroid latitude of a watershed
    - routing: 1 = traingular routing is involved | 0 = no routing
    
    hymod function Output:
    - qtotal: total flow/discharge
 
    '''
    #set model parameters
    kpwp = pars[0] # parameter to calculate permanent wilting point
    etexp = pars[1] # evapotranspiration reduction factor (when soil moisture is less than pwp)
    hmax = pars[2] # maximum storage/indicator height
    #bexp, power coefficient, can have values ranging from 0 to infinity, for calibration a transformation is applied 
    #which converts [0,inf] range of parameter to [0,2] so that search can be conducted on the finite range
    if pars[3] == 2: #to avoid undefined in log calculation
        bexp = 10**6 #a high value to represent infinity
    else:
        bexp = np.log(1 - pars[3] / 2) / np.log(0.5)    
    alpha = pars[4] #parameter which divides runoff into quick/direct flow and slow/base flow
    ks = pars[5] #resident time in slow release reservoir/ base flow reservoir
    lmax = pars[6] #maximum groundwater storage

    coeff_pet = pars[7] #coefficient for potential evapotranspiration

    ddf = pars[8] #degree-day factor
    scf = pars[9] #snowfall correction factor
    ts = pars[10] #threshold temperature for snow falling
    tm = pars[11] #threshold temperature for snowmelt
    tti = pars[12] #temperature interval for mixture of snow and rain
    whc = pars[13] #usually fixed, water holding capacity of snowpack (default 0.1)
    crf = pars[14] #usually fixed, refreezing coefficient (default 0.05)
    
    maxbas = pars[15] #traingular weighting function routing parameter, it represents time taken by water to move through the catchment and reach outlet

    #Initialize model variables
    sim_snow = np.zeros(len(p)) #simulated snow
    sim_swe =np.zeros(len(p)) #simulated snow water equivalent
    sim_melt = np.zeros(len(p)) #simulated snow melt
    pr_eff = np.zeros(len(p)) #effective precip (amount of liquid water available to enter soil matrix at a time step)

    state_soil = np.zeros(len(p)) #simulated soil moisuture storage accounting tank
    actevap = np.zeros(len(p)) #simulated actual evapotranspiration
    effrain = np.zeros(len(p)) #simulated effective rainfall
    direct = np.zeros(len(p)) #direct flow
    base = np.zeros(len(p)) #base flow

    soilwater = 0  #initial state of soil moisture storage
    groundwater = 100 #initial state of groundwater storage
    state_snow = 0 #initial state of snow storage
    state_sliq = 0 #initial state of liquid water on snowpack

    
    #Calculate potential evapotranspiration using Hamon's method
    #convert date to julian date
    date = pd.to_datetime(date) #convert first to python datetime format
    jdate = date.dt.strftime('%j').astype(int) #convert to julian date
    #calculate daylight hour
    var_theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.0086 * (jdate - 186)))
    var_pi = np.arcsin(0.39795 * np.cos(var_theta))
    daylighthr = 24 - 24 / math.pi * np.arccos((np.sin(0.8333 * math.pi / 180) + np.sin(latitude * math.pi / 180) * np.sin(var_pi)) / (np.cos(latitude * math.pi / 180) * np.cos(var_pi)))
    #now use Hamon's equation
    esat = 0.611 * np.exp(17.27 * temp/(237.3+temp))
    potevap = coeff_pet * 29.8 * daylighthr * (esat/(temp+273.2))

    
    ##-------Start of Time Loop-------
    for t in range(1, len(p)):
        
        ##Snow Routine
        '''
        Snow Routine takes temperature and precipitation as input and
        provides updated value of snow pack and water available to enter soil as output
        '''
        ct = temp[t] #temperature at current timestep
        cp = p[t] #precipitation at current timestep

        # Determine if precipitation is snow, rain, or a mixture
        if ct >= (ts + tti): #All rain, no snow
            snow = 0
            rain = cp
        elif ct <= ts:  # All snow, no rain
            snow = cp
            rain = 0
        else:  # Linear mixture of snow and rain in interval tti
            snowfrac = -1 / tti * (ct - ts) + 1
            snow = cp * snowfrac
            rain = cp * (1 - snowfrac)

        # If there is snow to melt
        if state_snow > 0:
            if ct > tm:
                melt = ddf * (ct - tm)
            else:
                melt = 0

            if melt > state_snow:
                # if melt>snow, add all snow and incoming rain to the liquid water storage in snow
                state_sliq += state_snow + rain
                state_snow = 0
            else:
                # otherwise, add melt portion and incoming rain to the liquid water storage in snow
                state_sliq += melt + rain
                state_snow -= melt

            # Calculate maximum liquid water held by the remaining snow
            liqmax = state_snow * whc

            if state_sliq > liqmax:
                pr_eff[t] = state_sliq - liqmax
                state_sliq = liqmax
            else:
                pr_eff[t] = 0
        else:
            melt = 0
            pr_eff[t] = rain

        # Calculate refreezing
        if ct < tm:
            refreeze = (tm - ct) * ddf * crf
        else:
            refreeze = 0

        if refreeze > state_sliq:
            # if refreeze >  liquid content of the snow, add entire liquid portion to current snow storage
            state_snow += state_sliq
            state_sliq = 0
        else: # if there is more liquid than will actually refreeze, add refreezing portion to the snow store
            state_snow += refreeze
            state_sliq -= refreeze

        # Final snow store for the time step by multiplying with snowfall correction factor
        state_snow += snow * scf

        sim_swe[t] = state_snow + state_sliq
        sim_snow[t] = snow
        sim_melt[t] = melt
        
        
        ##Soil Moisture Routine,HYMOD
        '''
        Soil moisture takes the available water to enter soil (pr_eff) as an input from snow module, updates
        soil moisture and recharge, and calculates direct runoff and base flow. 
        '''
        cmax = hmax / (1+bexp) #maximum storage capacity
        pwp = kpwp * cmax #permanent wilting point

        #make sure soilwater and groundwater are below their maximum value
        if soilwater > cmax:
            soilwater = cmax
        if groundwater > lmax:
            groundwater = lmax

        c_beg = soilwater #state of soil moisute storage at beginning of the time step
        #compute excess precipitation
        h_beg = hmax * (1 - (1- c_beg/cmax)**(1/ (1+bexp))) #Critical storage capacity based on the beginning state
        ov1 = max(0, pr_eff[t] + h_beg - hmax) #surface runoff
        inf = pr_eff[t] - ov1 #infiltrated water
        h_end = min(h_beg + inf, hmax) #updated critical storage capacity for current time step
        c_end = cmax * (1 - (1 - h_end/hmax)**(1+bexp)) #State of soil moisture account by the end of the time step for precipitation event
        ov2 = max(inf - (c_end - c_beg), 0) #Surface, Subsurface and groundwater runoff based on the probability-distributed principle

        #calculate potential evaporation value
        if sim_swe[t] > 0: #if snow no evap
            petval = 0
        else:
            petval = potevap[t]
        #calculate actual evaporation value
        if c_end >= pwp:
            kpet = 1
        else:
            kpet = (c_end / pwp) ** etexp
        et = kpet * petval 
        aet = min(c_end, et) #calculate actual evapotranspiration
        soilwater = c_end - aet #update state of soil moisture storage

        #partition ov1 and ov2 into quickflow and slowflow component
        directflow = ov1 + alpha * ov2 #effective rainfall which comprises direct runoff
        effloss = (1 - alpha) * ov2 #effective abstraction/loss contributing to groundwater storage

        #Check if ground water storage is full, if so no groundwater recharge
        if (effloss + groundwater) > lmax:
            directflow = directflow + (effloss + groundwater - lmax)
            effloss = lmax - groundwater
        
        #update groundwater storage and groundwater release
        update_gwater = (1 - ks)*groundwater + (1 - ks)*effloss
        baseflow = ks/(1-ks) * update_gwater
        groundwater = update_gwater

        #intermediate states
        state_soil[t] = soilwater + groundwater #State of soil moisture account at the beginning of time step
        actevap[t] = aet #Computed actual evaporation
        effrain[t] = ov1 + alpha * ov2 #Effective rainfall derived by probability-distributed principle

        #total outflow for the time step
        direct[t] = directflow
        base[t] = baseflow
    ##-------End of Time Loop-------
    

    ## Routing 
    if(routing == 1):
        #set integration step
        step = 0.005
        i = np.arange(0, maxbas + step, step)
        h = np.zeros(len(i))
        #define indices to construct traingular weighting function
        j = np.where(i<maxbas/2)
        h[j] = step * (i[j] *4 / maxbas ** 2)

        j = np.where(i >= maxbas/2)
        h[j] = step *(4 / maxbas - i[j] * 4 / maxbas **2)

        # Allow base of weighting function to be noninteger, adjust for extra weights for the last day
        if maxbas % 1 > 0:
            I = np.arange(1, len(i), (len(i) - 1) / maxbas)
            I = np.append(I, len(i))
        else:
            I = np.arange(1, len(i), (len(i) - 1) / maxbas)

        maxbas_w = np.zeros(len(I))

        # Integration of function
        for k in range(1, len(I)):
            maxbas_w[k] = np.sum(h[int(np.floor(I[k-1])):int(np.floor(I[k]))])

        # Ensure integration sums to unity for mass balance
        maxbas_w = maxbas_w[1:] / np.sum(maxbas_w[1:], where=~np.isnan(maxbas_w[1:]))

        # ROUTING OF DISCHARGE COMPONENTS
        qdirect = np.convolve(direct, maxbas_w, mode='full')[:len(p)]  # Routed direct flow
        qbase = np.convolve(base, maxbas_w, mode='full')[:len(p)]  # Routed base flow
        qtotal = qdirect + qbase
        
    else: #no routing
        qdirect = direct   #unrouted direct flow
        qbase = base   #unrouted base flow
        qtotal  = qdirect + qbase    #total flow 
    
    #return total flow as output
    return qtotal
#End of function  