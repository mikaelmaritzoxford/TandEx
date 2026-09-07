import numpy as np
from scipy.optimize import least_squares

# Constants
k = 1.380649e-23    # Boltzmann constant in J/K
q = 1.60217662e-19  # Electron charge in C
VSi_max = 0.77

def safe_exp_minus1(x, clip_high=700):
    # clip to avoid overflow
    return np.expm1(np.clip(x, -clip_high, clip_high))

def IP_func(x, T, params):
    # I = ILP - I0*(exp[(VP)/(nP*Vt)]-1) - (VP)/RshP
    Vt = (k * T) / q    # eV
    logI0P, nP, logI0Si, nSi, ILP, ILSi, RsP, RsSi, logRshP, logRshSi = params
    I, VP, VSi = x
    I0P = 10**logI0P
    RshP = 10**logRshP
    exp_term = safe_exp_minus1((VP+I*RsP)/(nP*Vt))
    return I - (ILP - I0P*exp_term - (VP+I*RsP)/RshP)

def ISi_func(x, T, params):
    # I = ILSi - I0*(exp[(VSi)/(nSi*Vt)]-1) - (VSi)/RshSi
    Vt = (k * T) / q    # eV
    logI0P, nP, logI0Si, nSi, ILP, ILSi, RsP, RsSi, logRshP, logRshSi = params
    I, VP, VSi = x
    I0Si = 10**logI0Si
    RshSi = 10**logRshSi
    exp_term = safe_exp_minus1((VSi+I*RsSi)/(nSi*Vt))
    return I - (ILSi - I0Si*exp_term - (VSi+I*RsSi)/RshSi)

def V_func(x, Vmeas, params):
    # VP + VSi + I*Rs = Vmeas
    logI0P, nP, logI0Si, nSi, ILP, ILSi, RsP, RsSi, logRshP, logRshSi = params
    I, VP, VSi = x
    return (VP + VSi) - Vmeas 

def IV_guess():
    I_guess = 20.5/1000
    VP_guess = 1.5
    VSi_guess = 0.65
    return [I_guess, VP_guess, VSi_guess]

def solve_tandem_IV_for_V(prev_guess, Vmeas, T, params, tol=1e-5, maxiter=150):

    # wrapper residual function expected by scipy.root
    def residuals_inner(x):
        # x is array-like [I, VP, VSi]
        r1 = IP_func(x, T, params)
        r2 = ISi_func(x, T, params)
        r3 = V_func(x, Vmeas, params)
        return np.array([r1, r2, r3], dtype=float)

    if prev_guess is None:
        x0 = IV_guess()
    else:
        x0 = prev_guess
    I_guess = x0[0]


    sol = least_squares(
        residuals_inner,
        x0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-6,
        max_nfev=maxiter,
        method="trf",
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.x

def I_model_from_V_array(V_array, T, params):
    I_model = np.empty_like(V_array, dtype=float)
    VP = np.empty_like(V_array, dtype=float)
    VSi = np.empty_like(V_array, dtype=float)
    prev_x = None
    for i, V in enumerate(V_array):
        try:
            sol = solve_tandem_IV_for_V(prev_x, V, T, params)
            I_model[i] = sol[0]
            VP[i] = sol[1]
            VSi[i] = sol[2]
            prev_x = sol
        except RuntimeError:
            I_model[i] = np.nan
            VP[i] = np.nan
            VSi[i] = np.nan
    return I_model, VP, VSi

def solve_tandem_for_I(I_target, T, params, prev_guess):
    def residuals(x):
        VP, VSi = x
        r1 = IP_func([I_target, VP, VSi], T, params)
        r2 = ISi_func([I_target, VP, VSi], T, params)
        return np.array([r1, r2])
    
    if prev_guess is None:
        I_guess, VP_guess, VSi_guess = IV_guess()
        x0 = [VP_guess, VSi_guess]
    else:
        x0 = prev_guess

    sol = least_squares(
        residuals, 
        x0, 
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-8,
    )

    logI0P, nP, logI0Si, nSi, ILP, ILSi, RsP, RsSi, logRshP, logRshSi = params
    VP, VSi = sol.x
    Vterm = VP + VSi #- I_target * Rs 
    return I_target, VP, VSi, Vterm

def V_model_from_I_array(I_targets, T, params):
    VP_vals = np.empty_like(I_targets, dtype=float)
    VSi_vals = np.empty_like(I_targets, dtype=float)
    Vterm_vals = np.empty_like(I_targets, dtype=float)
    prev_x = None

    for i, I_t in enumerate(I_targets):
        I_out, VP, VSi, Vterm = solve_tandem_for_I(I_t, T, params, prev_x)
        VP_vals[i] = VP
        VSi_vals[i] = VSi
        Vterm_vals[i] = Vterm
        prev_x = [VP, VSi]   # warm start for the next point

    return VP_vals, VSi_vals, Vterm_vals

def residuals(params, Imeas, Vmeas,T, penalty_weight = 100):   #0.786 gives Voc Si near 0.732
    Imodel,_,VSi= I_model_from_V_array(Vmeas, T, params)

    # penalty only when VSi exceeds the target ceiling
    penalty = np.sqrt(penalty_weight) * np.maximum(0.0, VSi - VSi_max)**2

    residuals_current = Imodel - Imeas
        
    return np.concatenate([residuals_current, penalty])


from scipy.optimize import brentq

#def Voc_Si_from_params(params, T):
#    _, VP_oc, VSi_oc, Vterm_oc = solve_tandem_for_I(0.0, T, params, prev_guess=None)
#    return VSi_oc
#
#
#def residuals(params, Imeas, Vmeas, T, VocSi_max=0.7443, penalty_weight=20):
#    Imodel, _, _ = I_model_from_V_array(Vmeas, T, params)
#
#    residuals_current = Imodel - Imeas
#
#    VocSi_model = Voc_Si_from_params(params, T)
#    penalty_voc = np.sqrt(penalty_weight) * np.maximum(0.0, VocSi_model - VocSi_max)
#
#    return np.concatenate([residuals_current, np.atleast_1d(penalty_voc)])



# initial guess
def initial_guess():
    # I0 guesses: small values
    logI0Si_0 = -14
    logI0P_0 = -12
    # ideality guesses
    nP_0 = 1.8
    nSi_0 = 1.3
    ILP = 21/1000 #convert to mA
    ILSi = 20.5/1000
    RsP = 5
    RsSi = 0.2
    logRshP = 4
    logRshSi = 5
    return [logI0P_0, nP_0, logI0Si_0, nSi_0, ILP, ILSi, RsP, RsSi, logRshP, logRshSi]

# bounds
def bounds():
    # I0P, nP, I0Si, nSi, Iph/IL, Rs
    lb = np.array([-17, 1.4, -18, 0.8, 19/1000, 19/1000, 0.01, 0.01, 3, 4])
    ub = np.array([-8, 2.5, -10, 1.8, 22/1000, 22/1000, 12, 2, 5, 7])
    return lb, ub

def fit_curve(Vmeas, Imeas, T, optimize = False, verbose = True):

    order = np.argsort(Vmeas)
    Vmeas, Imeas = Vmeas[order], Imeas[order]

    x0 = initial_guess()
    lb, ub = bounds()

    if verbose ==True:
        verbose = 2

    if optimize:
        from scipy.optimize import differential_evolution

        # Scalar objective for global optimizers
        def scalar_objective(params):
            r = residuals(params, Imeas, Vmeas, T)
            return np.nansum(r**2)

        # Global search (coarse)
        result_global = differential_evolution(
            scalar_objective, 
            bounds=list(zip(lb, ub)),
            maxiter=20,
            polish=False,  # we'll polish ourselves
            seed=42
        )

        # Local refinement (fine)
        res = least_squares(
            residuals, 
            result_global.x, 
            args=(Imeas, Vmeas, T),
            bounds=(lb, ub)
        )

    if not optimize:
        res = least_squares(
            residuals, 
            x0, 
            args=(Imeas,Vmeas,T), 
            bounds=(lb, ub), 
            xtol=1e-13, 
            ftol=1e-13, 
            gtol=1e-8, 
            max_nfev=200, 
            verbose=verbose
        )

    print(f'VSi_max = {VSi_max:.5g}')
    return res

from scipy.interpolate import interp1d
def findMpp(V,I):
    P = V*I
    idx = np.argmax(P)
    return I[idx], V[idx], P[idx]

def findVoc(V,I):
    f = interp1d(I,V)
    return float(f(0.0))
        
def findIsc(V,I):
    f = interp1d(V,I)
    return float(f(0.0))
        
def findFF(Pmpp, voc, isc):
    return Pmpp/(isc*voc)

def findPerformancePoints(V, I):
    Impp, Vmpp, Pmpp = findMpp(V, I)
    isc = findIsc(V, I)
    voc = findVoc(V, I)
    FF = findFF(Pmpp, voc, isc)
    return isc, voc, Impp, Vmpp, Pmpp, FF

def printPerformancePoints(Vmeas, Imeas, res, T):
    isc_meas, voc_meas, Impp_meas, Vmpp_meas, Pmpp_meas, FF_meas = findPerformancePoints(Vmeas, Imeas)
    print("\nMeasured Remarkable Points:")
    print(f"isc_meas = {isc_meas:.5g} A")
    print(f"voc_meas = {voc_meas:.5g} V")
    print(f"Impp_meas = {Impp_meas:.5g} A")
    print(f"Vmpp_meas = {Vmpp_meas:.5g} V")
    print(f"Pmpp_meas = {Pmpp_meas:.4g} W")
    print(f"FF_meas = {FF_meas:.4g}")

    Vfit = np.linspace(min(Vmeas), max(Vmeas), 400)
    fitted_params = res.x
    Ifit,_,_ = I_model_from_V_array(Vfit, T, fitted_params)

    isc_fit, voc_fit, Impp_fit, Vmpp_fit, Pmpp_fit, FF_fit = findPerformancePoints(Vfit, Ifit)

    Vfit = np.linspace(min(Vmeas), max(Vmeas), 400)
    fitted_params = res.x
    Ifit,_,_ = I_model_from_V_array(Vfit, T, fitted_params)

    isc_fit, voc_fit, Impp_fit, Vmpp_fit, Pmpp_fit, FF_fit = findPerformancePoints(Vfit, Ifit)
    print("\nFitted Remarkable Points:")
    print(f"isc_fit = {isc_fit:.5g} A")
    print(f"voc_fit = {voc_fit:.5g} V")
    print(f"Impp_fit = {Impp_fit:.5g} A")
    print(f"Vmpp_fit = {Vmpp_fit:.5g} V")
    print(f"Pmpp_fit = {Pmpp_fit:.4g} W")
    print(f"FF_fit = {FF_fit:.4g}")

def findError(Vmeas, Imeas, T, fitted_params):
    Imodel,_,_ = I_model_from_V_array(Vmeas, T,  fitted_params)
    meanSquaredError = ((Imodel - Imeas) ** 2).mean()
    rmse = np.sqrt(meanSquaredError)
    current_range = np.max(Imeas) - np.min(Imeas)
    nrmse = rmse/current_range*100
    return rmse, nrmse

import pandas as pd
def get_IV(path):
    IV = pd.read_csv(path)
    V = IV['V']
    I = IV['I']
    return np.asarray(V), np.asarray(I)
