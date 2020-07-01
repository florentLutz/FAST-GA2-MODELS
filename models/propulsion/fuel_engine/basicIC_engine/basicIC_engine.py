# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04 10:47:55 2018

@author: hermetz
    ICE for Internal Combustion Engine or pistion engine
    4 strokes gasoline firstly implemented
    Potential upgrade : 
        2 or 4 stokes, gasoline or diesel
        atmospheric, compressed/turbocompressed

"""

from fast.atmosphere import atmosphere

class BasicICEngine(object):   

    def __init__(self,  Psls, fuel_type, n_strokes):
    # def __init__(self,  Psls): #modified NRG
        self.Psls=Psls
#        self.temperature = None
        self.altitude = None
#        self.density = None
        self.Mach = None
        self.fuel_type = fuel_type #modified NRG
        self.n_strokes = n_strokes #modified NRG
        
    def compute_Psmax(self, altitude):
        
        sigma=atmosphere(altitude)[1]/atmosphere(0)[1]
        
        # Calcul de puissance max disponible pour Z et V
        # Psls in kW
        Ps_max = self.Psls*(sigma-(1-sigma)/7.55)

        return Ps_max
        

    def compute_manual(self, Mach, altitude, thrust_rate, phase):
        """
        #----------------------------------------------------------------
        # DEFINITION OF THE available Power and Fuel flow
        #----------------------------------------------------------------
        # from ONERA ICE regression analysis - Only 4 strokes Gasoline engine
        #
        #----------------------------------------------------------------
        # INPUTS
        #    -mach number
        #    -Flight altitude [m]
        #    -thrust rate : 0,93 for ampere
        #    -phase; no interest here, just to fit to the code 
        
        # Mach number not yet used as potentially required for t
        # OUTPUTS
        #    -sfc in [kg/s/N]urbocharged engine
        #
        #    -available shaft power Ps in [W]  
        #----------------------------------------------------------------
        """
        self.Mach = Mach

        # Calcul de puissance max disponible pour Z et V
        # Psls in kW
        Ps_max = self.compute_Psmax(altitude)
        #Passage de P en WATT et avec regime reduit
        Ps = thrust_rate *Ps_max*1000
        
        
        # Calcul de conso specifique a poussee max disponible Moteur gasoline 2 et 4 temps
        if self.fuel_type ==1.:
                if self.n_strokes ==2.:    # Gasoline engine 2 strokes
                    sfc_max = 1125.9*Ps_max**(-0.2441)
                elif self.n_strokes ==4.:  # Gasoline engine 4 strokes
                    sfc_max = -0.0011*Ps_max**2  + 0.5905*Ps_max + 228.58 
                else:
                    print 'Bad configuration: Unknown n_strokes %d', self.n_strokes
                    raise RuntimeError()
        elif self.fuel_type ==2.:          
                if self.n_strokes ==2.:    # Diesel engine 2 strokes
                    sfc_max = 0 # To complete with the needed ec
                elif self.n_strokes ==4.:  # Diesel engine 4 strokes
                    sfc_max = 0 # To complete with the needed ec
                else:
                    print 'Bad configuration: Unknown n_strokes %d', self.n_strokes
                    raise RuntimeError()
        else:
                print 'Bad configuration: Unknown fuel_type %d', self.fuel_type
                raise RuntimeError()  
        

        # Calcul de conso specifique en regime reduit
        # TBD - No model available, conservative solution
        sfc = sfc_max*(-0.9976*thrust_rate**2 + 1.9964*thrust_rate)
        sfc = sfc/1e6/3600.  # kg.W.s

        return Ps, sfc
        
    def compute_regulated(self, Mach, altitude,drag):
        
        """
        #----------------------------------------------------------------
        # DEFINITION OF THE SFC related to the curent flight condition
        #----------------------------------------------------------------
        # from ONERA ICE regression analysis - Only 4 strokes Gasoline engine
        #
        #----------------------------------------------------------------
        # INPUTS
        #    -mach number
        #    -Flight altitude [m]
        #    -Flight temperature [K]
        #
        # Mach number not yet used as potentially required for turbocharged engine
        #
        # OUTPUTS
        #    -throttle %
        #    -sfc in [kg/s/N]
        #    
        #----------------------------------------------------------------
        """
    
        self.Mach = Mach        
        V = Mach* atmosphere(altitude)[4]
        Ps_max = self.compute_Psmax(altitude)  ###Gagg and Ferror formula, Psls (kW)   
        requiredPower = drag* V # in W
        throttle = requiredPower/(Ps_max*1000) #Passe Ps_max en W

        # Calcul de conso specifique a poussee max disponible Moteur gasoline 2 et 4 temps
        if self.fuel_type ==1.:
                if self.n_strokes ==2.:    # Gasoline engine 2 strokes
                    sfc_max = 1125.9*Ps_max**(-0.2441)
                elif self.n_strokes ==4.:  # Gasoline engine 4 strokes
                    sfc_max = -0.0011*Ps_max**2  + 0.5905*Ps_max + 228.58 
                else:
                    print 'Bad configuration: Unknown n_strokes %d', self.n_strokes
                    raise RuntimeError()
        elif self.fuel_type ==2.:          
                if self.n_strokes ==2.:    # Diesel engine 2 strokes
                    sfc_max = 0 # To complete with the needed ec
                elif self.n_strokes ==4.:  # Diesel engine 4 strokes
                    sfc_max = 0 # To complete with the needed ec
                else:
                    print 'Bad configuration: Unknown n_strokes %d', self.n_strokes
                    raise RuntimeError()
        else:
                print 'Bad configuration: Unknown fuel_type %d', self.fuel_type
                raise RuntimeError()     
        # Calcul de conso specifique en regime reduit
        # TBD
        sfc = sfc_max*(-0.9976*throttle**2 + 1.9964*throttle)
        
        if requiredPower <= Ps_max*1000.:
            sfc = sfc_max/1e6/3600.                           
        else:                                                  ## value given is 7.70496343035e-08, 
            print 'available Power lower than required !! Power available :', Ps_max
            print 'required::',requiredPower/1000
            raise RuntimeError()
#        print 'Pmax a cette alt',Ps_max
#        print 'Putile',requiredPower/1000
        return  sfc, throttle   # modified NRG
        
        
    def sizing(self, Power):
        
        
        Power_diesel = [81, 87, 118, 123.5, 132, 160, 169, 194,210,221,228,239,250,257, 368]
        Length_diesel = [650, 635, 861,738, 738, 859, 834, 834, 900, 980, 900, 900, 900,900, 1114]
        Height_diesel = [580,434,655,574,574,659,750,784,770,770,770,770,770,770,712]
        Width_diesel = [740,641,585,855,855,650,750,931,670,670,670,670,670,670,850]
                
        
        return Power
        
        
class Propeller(object):          
    def __init__(self,eff_max, dia, blades):
        self.dia = dia                  
        self.blades = blades               
        self.eff_max=eff_max

    def compute_efficiency(self, mach):
        
#        self.altitude = altitude * 0.3048
        self.mach = mach
#        self.temperature, self.density, _, _ = self._atmosphere(self.altitude)        
        if mach<0.1:
            eff=10*mach*self.eff_max
        elif mach <= 0.7:
            eff=self.eff_max
        else:
            eff=(1-(mach-0.7)/3)*self.eff_max
        return eff
        
      