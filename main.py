from Inviscid_Burgers import *

#Initialize object of the class with required specifications
domains = domain(20,0.01,-1.0,1.0,duration = 0.5)

domains.GudonovMethod()
domains.FluxSplittingMethod() 
domains.FluxDifferenceSplittingMethod()