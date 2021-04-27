from importlib import resources as rsc
from ProteusTools.protmc import resources

with rsc.path(resources, 'protMC.exe') as path:
    PROTMC_PATH = str(path)
with rsc.path(resources, 'ADAPT.conf') as path:
    ADAPT_CONF_PATH = str(path)
with rsc.path(resources, 'MC.conf') as path:
    MC_CONF_PATH = str(path)
