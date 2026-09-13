import numpy as np
import lumicks.pylake as lk
import sys
from scipy.integrate import cumulative_trapezoid

#%%
sys.path.insert(1, '/Users/gabriel/scripts/optical_tweezers_toolbox')
from tweezers_toolbox_modules.general_utils import get_KbT
from tweezers_toolbox_modules.general_utils import FastInverseModel

#%% FUNCTIONS

def peeling_rigid_body_contribution(x, R, units="nm"):
    w = (1.65 * 2 * np.pi) / 145

    if units == "nm":
        d = 2 * R * np.cos((w / (2 * 0.34)) * x)

    elif units == "bp":
        d = 2 * R * np.cos((w / 2) * x)

    else:
        raise ValueError("units must be 'nm' or 'bp'")

    return d

def bell_rate_model (F, k0, x_dagger, T = 298.15):
    """
    Simple bell model force-dependent unfolding rate.

    F          : force in pN
    k0         : zero-force rate in s^-1
    x_dagger   : zero-force distance to transition state in nm
    T          : temperature in K
    """
    kBT = get_KbT(T) # 1 kBT is 4.1164 pN*nm
    k = k0*np.exp(F*x_dagger/kBT)
    
    return(k)
    
def CHS_rate_model(F, k0, x_dagger, dG_dagger, nu=0.5, T=298.15):
    """
    Cossio-Hummer-Szabo force-dependent unfolding rate.

    F          : force in pN
    k0         : zero-force rate in s^-1
    x_dagger   : zero-force distance to transition state in nm
    dG_dagger  : zero-force barrier in units of kBT
    nu         : landscape-shape parameter
    T          : temperature in K
    """

    kB = 0.01380649  # pN nm / K
    kBT = kB * T

    z = 1 - nu * F * x_dagger / (dG_dagger * kBT)

    k = (
        k0
        * z**(2 - 1/nu)
        * np.exp(
            dG_dagger * (1 - z**(1/nu))
        )
    )

    return k

def rupture_force_pdf_cdf_ccdf(F, kF, Fdot):

    # k(F) / Fdot(F)
    k_over_Fdot = kF / Fdot

    # Integral: ∫[0→F] k(f) / Fdot(f) df
    integral = cumulative_trapezoid(
        k_over_Fdot,
        F,
        initial=0
    )

    # Survival:
    # S(F) = exp[-∫[0→F] k(f) / Fdot(f) df]
    survival = np.exp(-integral)

    # Rupture-force PDF:
    # p(F) = [k(F) / Fdot(F)] * S(F)
    p_F = pdf = k_over_Fdot * survival
    P_F = 1-survival

    return p_F,P_F, survival

def ini_eWLC(
    domain_type,
    domain,
    *args,
    inversion_method="fast"
):
    """
    Use GenWLCModel to build the (e)WLC model and evaluate it.

    domain_type:
      0: distance -> force (invert)
      1: force    -> distance (forward)

    *args:
      one or more component tuples:
      (Lp, Lc) or (Lp, Lc, St)

    inversion_method:
      "fast"   -> vectorized safeguarded Newton inversion
      "pylake" -> original Pylake inversion
    """

    if not args:
        raise ValueError(
            "No components provided. "
            "Pass at least one (Lp, Lc[, St]) tuple."
        )

    gm = GenWLCModel(
        *args,
        eWLC_approx="marko"
    )

    gm.gen_model()
    gm.gen_par_dict()

    # --------------------------------
    # distance -> force
    # --------------------------------
    if domain_type == 0:

        if inversion_method == "fast":

            inv_model = FastInverseModel(
                gm.model
            )

        elif inversion_method == "pylake":

            inv_model = gm.model.invert()

            # In case invert() modifies in place
            if inv_model is None:
                inv_model = gm.model

        else:
            raise ValueError(
                "inversion_method must be "
                "'fast' or 'pylake'."
            )

        return inv_model(
            domain,
            gm.pardict
        )

    # --------------------------------
    # force -> distance
    # --------------------------------
    elif domain_type == 1:

        return gm.model(
            domain,
            gm.pardict
        )

    else:
        raise ValueError(
            "domain_type must be 0 "
            "(distance->force) or 1 "
            "(force->distance)."
        )

## WORKING ON THIS
def ini_eFJC(domain_type, domain, *args):
    """
    Use GenWLCModel to build the (e)FJC model and evaluate it.

    domain_type:
      0: distance -> force (invert)
      1: force    -> distance (forward)

    *args: one or more component tuples:
      (Lp, Lc) or (Lp, Lc, St)
    """
    return()
##

def WLC_rms_endtoend_dist (L0, LP):
    R2_ens_avr = 2*LP*L0*( 1- ((LP/L0) * (1-np.exp(-L0/LP))) )
    R_ens_avg = np.sqrt(R2_ens_avr)
    return(R_ens_avg)


def get_trap_plus_handles_stiffness_num(F, trap_keq, *eWLC_pars):
    F = np.asarray(F, dtype=float)

    if F.ndim != 1:
        raise ValueError("F must be a 1D array.")

    if len(F) < 2:
        raise ValueError("F must contain at least 2 points.")

    if not np.all(np.isfinite(F)):
        raise ValueError("F contains NaN or inf values.")

    if trap_keq <= 0:
        raise ValueError("trap_keq must be positive.")

    px_handle = ini_eWLC(1, F, *eWLC_pars)

    if not np.all(np.isfinite(px_handle)):
        raise ValueError("eWLC model returned NaN or inf values.")

    if np.any(np.diff(px_handle) == 0):
        raise ValueError("eWLC extension contains repeated adjacent values.")

    k_handles_F = np.gradient(F, px_handle)

    ktot_F = (1/trap_keq + 1/k_handles_F)**-1

    return ktot_F

def get_loading_rate(F, pulling_speed, trap_keq, *eWLC_pars):

    if not np.isscalar(pulling_speed):
        raise ValueError("pulling_speed must be a scalar.")

    if not np.isfinite(pulling_speed):
        raise ValueError("pulling_speed must be finite.")

    trap_keq = np.asarray(trap_keq, dtype=float)

    # One trap stiffness
    if trap_keq.ndim == 0:

        ktot_F = get_trap_plus_handles_stiffness_num(
            F, trap_keq, *eWLC_pars
        )

        Fdot = ktot_F * pulling_speed

    # One trap stiffness per trace
    elif trap_keq.ndim == 1:

        Fdot = []

        for keq in trap_keq:

            ktot_F = get_trap_plus_handles_stiffness_num(
                F, keq, *eWLC_pars
            )

            Fdot.append(ktot_F * pulling_speed)

        Fdot = np.array(Fdot)

    else:
        raise ValueError("trap_keq must be a scalar or 1D array.")

    return Fdot

#%% CLASSES

class GenWLCModel:
    def __init__(self, *WLCpars, fixed_parameter_mask=None,
                 distance=None, force=None,
                 distance_offset=False, force_offset=False,
                 eWLC_approx="marko", global_fit_obj=False):
        self.WLCpars = list(WLCpars)
        self.model = None
        self.distance = distance
        self.force = force
        self.fitobj = None
        self.distance_offset = distance_offset
        self.force_offset = force_offset
        self.eWLC_approx = eWLC_approx
        self.global_fit_obj = global_fit_obj
        self.fixed_parameter_mask = fixed_parameter_mask  # validated only when gen_fitobj() is called

    def gen_model(self):
        for c, pars in enumerate(self.WLCpars):
            comp = self._make_component(c, pars)
            if c == 0:
                if self.force_offset:
                    comp = comp.subtract_independent_offset()

                if self.distance_offset:
                    comp = comp + lk.distance_offset("component_0")
                    
                self.model = comp
            else:
                self.model = self.model + comp

    def gen_par_dict(self):
        self.pardict = {}
        for c, pars in enumerate(self.WLCpars):
            if len(pars) == 2:
                self.pardict[f"component_{c}/Lp"] = pars[0]
                self.pardict[f"component_{c}/Lc"] = pars[1]
            elif len(pars) == 3:
                self.pardict[f"component_{c}/Lp"] = pars[0]
                self.pardict[f"component_{c}/Lc"] = pars[1]
                self.pardict[f"component_{c}/St"] = pars[2]
            else:
                raise ValueError("Please provide valid parameter lists.")
        self.pardict["kT"] = 4.11

    def gen_fitobj(self):
        # validate mask FIRST (required here)
        self._check_fixed_parameter_mask()

        # create fit object
        self.fitobj = lk.FdFit(self.model)

        # add data
        # Note: Here, force always goes in front of distance. Lumicks add_data method
        # will handle which variable is dependent or independent based on
        # object property of the model.
        if self.global_fit_obj:
            for i, (d, f) in enumerate(zip( self.distance,self.force)):
                if i == 0:
                    self.fitobj.add_data(f"AdK {i}", f, d)
                else:
                    if self.force_offset and self.distance_offset:
                        self.fitobj.add_data(
                            f"AdK {i}", f, d,
                            params={
                                "component_0/f_offset": f"component_0/f_offset_{i}",
                                "component_0/d_offset": f"component_0/d_offset_{i}"
                            }
                        )
                    elif self.force_offset:
                        self.fitobj.add_data(
                            f"AdK {i}", f, d,
                            params={"component_0/f_offset": f"component_0/f_offset_{i}"}
                        )
                    elif self.distance_offset:
                        self.fitobj.add_data(
                            f"AdK {i}", f, d,
                            params={"component_0/d_offset": f"component_0/d_offset_{i}"}
                        )
                    else:
                        self.fitobj.add_data(f"AdK {i}", f, d)
        else:
            self.fitobj.add_data("", self.force, self.distance)

        # set starting values
        for c, pars in enumerate(self.WLCpars):
            if len(pars) == 2:
                self.fitobj[f"component_{c}/Lp"].value = pars[0]
                self.fitobj[f"component_{c}/Lc"].value = pars[1]
            elif len(pars) == 3:
                self.fitobj[f"component_{c}/Lp"].value = pars[0]
                self.fitobj[f"component_{c}/Lc"].value = pars[1]
                self.fitobj[f"component_{c}/St"].value = pars[2]
            else:
                raise ValueError("Please provide valid parameter lists.")

        # apply free/fixed flags (mask already validated)
        for c, pars in enumerate(self.WLCpars):
            mask = self.fixed_parameter_mask[c]
            if len(pars) == 2:
                names = ("Lp", "Lc")
            elif len(pars) == 3:
                names = ("Lp", "Lc", "St")
            else:
                raise ValueError("Please provide valid parameter lists.")
            for i, name in enumerate(names):
                self.fitobj[f"component_{c}/{name}"].fixed = bool(mask[i])

    def _make_component(self, c, pars):
        """Build a single component model for component_{c} based on len(pars) and approx."""
        name = f"component_{c}"
        if len(pars) == 2:
            return lk.wlc_marko_siggia_distance(name)
        elif len(pars) == 3:
            if self.eWLC_approx == "marko":
                return lk.ewlc_marko_siggia_distance(name)
            elif self.eWLC_approx == "odijk":
                return lk.ewlc_odijk_distance(name)
            raise ValueError("Please select a valid approx (marko/odijk)")
        raise ValueError("Please provide valid parameter lists.")
        
    def _check_fixed_parameter_mask(self):
        """Validate self.fixed_parameter_mask strictly. Raises on any mismatch; returns nothing."""
        fm = self.fixed_parameter_mask
        if fm is None:
            raise ValueError("fixed_parameter_mask is required when calling gen_fitobj().")
        if not isinstance(fm, (list, tuple)):
            raise TypeError("fixed_parameter_mask must be a list/tuple of per-component lists/tuples of booleans.")
        if len(fm) != len(self.WLCpars):
            raise ValueError(f"fixed_parameter_mask has {len(fm)} entries but WLCpars has {len(self.WLCpars)} components.")

        for c, pars in enumerate(self.WLCpars):
            if len(pars) == 2:
                names = ("Lp", "Lc")
            elif len(pars) == 3:
                names = ("Lp", "Lc", "St")
            else:
                raise ValueError("Please provide valid parameter lists (2 or 3 elements).")

            m = fm[c]
            if not isinstance(m, (list, tuple)):
                raise TypeError(f"fixed_parameter_mask[{c}] must be a list/tuple.")
            if len(m) != len(names):
                raise ValueError(
                    f"fixed_parameter_mask[{c}] length {len(m)} does not match number of parameters {len(names)} "
                    f"for component_{c} {names}."
                )
            for x in m:
                if not isinstance(x, (bool, int)):  # allow 0/1
                    raise TypeError(f"fixed_parameter_mask[{c}] must contain only booleans (or 0/1).")

#%% DEPRECATED

def ini_eWLC_old2(domain_type,domain, *args):    
    args = list(args)
    model = None
        
    # construct the model
    for c, pars in enumerate(args):
        if c == 0:
            if len(pars) == 2:
                model = lk.wlc_marko_siggia_distance(f"component_{c}")
            elif len(pars) == 3:
                model = lk.ewlc_marko_siggia_distance(f"component_{c}")
            else:
                return("Please provide valid parameter lists.")
        else:
            if len(pars) == 2:
                model = model + lk.wlc_marko_siggia_distance(f"component_{c}")
            elif len(pars) == 3:
                model = model + lk.ewlc_marko_siggia_distance(f"component_{c}")
            else:
                return("Please provide valid parameter lists.")
    
    # Generate parameter's dict
    pardict = {}
    
    for c, pars in enumerate(args):
        if len(pars) == 2:
            pardict[f"component_{c}/Lp"] = pars[0]
            pardict[f"component_{c}/Lc"] = pars[1]
        elif len(pars) == 3:
            pardict[f"component_{c}/Lp"] = pars[0]
            pardict[f"component_{c}/Lc"] = pars[1]
            pardict[f"component_{c}/St"] = pars[2]
        else:
            return("Please provide valid parameter lists.")
    
    pardict["kT"] = 4.11
    
    # predict
    if domain_type == 0:
        model = model.invert()
        pdis = domain
        pF = model(pdis, pardict)
        return(pF)
    else:
        if domain_type == 1:
            pF = domain
            pdis = model(pF, pardict)
            return(pdis)
        
def ini_eWLC_old (domain_type,domain, eWLC_par_dna, eWLC_par_rna):
    Lp_dna = eWLC_par_dna[0]
    Lc_dna = eWLC_par_dna[1]
    St_dna = eWLC_par_dna[2]

    Lp_rna = eWLC_par_rna[0]
    Lc_rna = eWLC_par_rna[1]
    St_rna = eWLC_par_rna[2]

    model = lk.ewlc_odijk_distance("DNA") + lk.ewlc_odijk_distance("RNA")

    if domain_type == 0:
        model = model.invert()
        pdis = domain
        pF = model(pdis, {"DNA/Lp": Lp_dna,
                          "DNA/Lc": Lc_dna,
                          "DNA/St": St_dna,
                          "kT": 4.11,
                          "RNA/Lp": Lp_rna,
                          "RNA/Lc": Lc_rna,
                          "RNA/St": St_rna})
        return(pF)
    else:
        if domain_type == 1:
            pF = domain
            pdis = model(pF, {"DNA/Lp": Lp_dna,
                              "DNA/Lc": Lc_dna,
                              "DNA/St": St_dna,
                              "kT": 4.11,
                              "RNA/Lp": Lp_rna,
                              "RNA/Lc": Lc_rna,
                              "RNA/St": St_rna})
            return(pdis)
        
def simple_ini_eWLC (domain_type,domain, eWLC_par_dna):
    
    Lp_dna = eWLC_par_dna[0]
    Lc_dna = eWLC_par_dna[1]
    St_dna = eWLC_par_dna[2]


    model = lk.ewlc_marko_siggia_distance("DNA") 

    if domain_type == 0:
        model = model.invert()
        pdis = domain
        pF = model(pdis, {"DNA/Lp": Lp_dna,
                          "DNA/Lc": Lc_dna,
                          "DNA/St": St_dna,
                          "kT": 4.11})
        return(pF)
    else:
        if domain_type == 1:
            pF = domain
            pdis = model(pF, {"DNA/Lp": Lp_dna,
                              "DNA/Lc": Lc_dna,
                              "DNA/St": St_dna,
                              "kT": 4.11})
            return(pdis)

class GenWLCModel_old:
    def __init__(self, *WLCpars, distance = None, force = None, var_component = 0,
                 lc_bounds = [0.001, 100], distance_offset = False, force_offset = False,
                 approx = "marko"):
        self.WLCpars = list(WLCpars)
        self.model = None
        self.errcod = 1
        self.distance = distance
        self.force = force
        self.fitobj = None
        self.var_component = var_component
        self.lc_bounds = lc_bounds
        self.distance_offset = distance_offset
        self.force_offset = force_offset
        self.approx = approx
    
    def gen_model(self):
        for c, pars in enumerate(self.WLCpars):
            if c == 0:
                if self.force_offset:
                    if len(pars) == 2:
                        self.model = lk.wlc_marko_siggia_distance(f"component_{c}").subtract_independent_offset()
                    elif len(pars) == 3:
                        if self.approx == "marko":
                            self.model = lk.ewlc_marko_siggia_distance(f"component_{c}").subtract_independent_offset()
                        elif self.approx == "odijk":
                            self.model = lk.ewlc_odijk_distance(f"component_{c}").subtract_independent_offset()
                        else:
                            self.errcod = 0
                            return("Please select a valid approx (marko/odijk)")
                    else:
                        self.errcod = 0
                        return("Please provide valid parameter lists.")
                    
                else:
                    if len(pars) == 2:
                        self.model = lk.wlc_marko_siggia_distance(f"component_{c}")
                    elif len(pars) == 3:
                        if self.approx == "marko":
                            self.model = lk.ewlc_marko_siggia_distance(f"component_{c}")
                        elif self.approx == "odijk":
                            self.model = lk.ewlc_odijk_distance(f"component_{c}")
                        else:
                            self.errcod = 0
                            return("Please select a valid approx (marko/odijk)")
                    else:
                        self.errcod = 0
                        return("Please provide valid parameter lists.")
                
                if self.distance_offset:
                   self.model = self.model + lk.distance_offset("component_0")
                 
            else:
                if len(pars) == 2:
                    self.model = self.model + lk.wlc_marko_siggia_distance(f"component_{c}")
                elif len(pars) == 3:
                    if self.approx == "marko":
                        self.model = self.model + lk.ewlc_marko_siggia_distance(f"component_{c}")
                    elif self.approx == "odijk":
                        self.model = self.model + lk.ewlc_odijk_distance(f"component_{c}")
                    else:
                        self.errcod = 0
                        return("Please select a valid approx (marko/odijk)")
                else:
                    self.errcod = 0
                    return("Please provide valid parameter lists.")
    
    def gen_par_dict(self):
        # Generate parameter's dict
        self.pardict = {}
        
        for c, pars in enumerate(self.WLCpars):
            if len(pars) == 2:
                self.pardict[f"component_{c}/Lp"] = pars[0]
                self.pardict[f"component_{c}/Lc"] = pars[1]
            elif len(pars) == 3:
                self.pardict[f"component_{c}/Lp"] = pars[0]
                self.pardict[f"component_{c}/Lc"] = pars[1]
                self.pardict[f"component_{c}/St"] = pars[2]
            else:
                self.errcod = 0
                return("Please provide valid parameter lists.")
        self.pardict["kT"] = 4.11
    
    def gen_fitobj(self):
        # Creating fit object with data
        #print(self.model)
        self.fitobj = lk.FdFit(self.model)
        self.fitobj.add_data("", self.force, self.distance)
        
        #define values
        for c, pars in enumerate(self.WLCpars):
            if len(pars) == 2:
                self.fitobj[f"component_{c}/Lp"].value = pars[0]
                self.fitobj[f"component_{c}/Lc"].value = pars[1]
            elif len(pars) == 3:
                self.fitobj[f"component_{c}/Lp"].value = pars[0]
                self.fitobj[f"component_{c}/Lc"].value = pars[1]
                self.fitobj[f"component_{c}/St"].value = pars[2]
            else:
                self.errcod = 0
                return("Please provide valid parameter lists.")
        
        # define is par is floating
        for c, pars in enumerate(self.WLCpars):
            if len(pars) == 2:
                self.fitobj[f"component_{c}/Lp"].fixed = True
                self.fitobj[f"component_{c}/Lc"].fixed = True                    
            elif len(pars) == 3:
                self.fitobj[f"component_{c}/Lp"].fixed = True
                self.fitobj[f"component_{c}/Lc"].fixed = True
                self.fitobj[f"component_{c}/St"].fixed = True
            else:
                self.errcod = 0
                return("Please provide valid parameter lists.")
            
        # Free float Lc of variable component
        self.fitobj[f"component_{self.var_component}/Lc"].fixed = False
        
        # Define bounds
        self.fitobj[f"component_{self.var_component}/Lc"].lower_bound = self.lc_bounds[0]
        self.fitobj[f"component_{self.var_component}/Lc"].upper_bound = self.lc_bounds[1] 