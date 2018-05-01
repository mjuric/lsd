# 2D dust maps
from dustmaps.sfd import SFDQuery
from dustmaps.planck import PlanckQuery
from dustmaps.lenz2017 import Lenz2017Query

# 3D dust maps
from dustmaps.bayestar import BayestarQuery
from dustmaps.marshall import MarshallQuery
from dustmaps.iphas import IPHASQuery

# Lazily create dust maps
from ..utils import LazyCreate

SFD = LazyCreate(SFDQuery)
Planck = LazyCreate(PlanckQuery)
PlanckTau = LazyCreate(PlanckQuery, component='tau')
PlanckR = LazyCreate(PlanckQuery, component='radiance')
#PlanckT = LazyCreate(PlanckQuery, component='temperature')
PlanckBeta = LazyCreate(PlanckQuery, component='beta')
Lenz2017 = LazyCreate(Lenz2017Query)

Bayestar2017 = LazyCreate(BayestarQuery, version='bayestar2017')
Bayestar2015 = LazyCreate(BayestarQuery, version='bayestar2015')
Marshall = LazyCreate(MarshallQuery)
IPHAS = LazyCreate(IPHASQuery)
