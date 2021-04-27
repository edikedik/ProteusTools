from importlib import resources

with resources.path('resources', 'protMC.exe') as path:
    PROTMC_PATH = path.name
with resources.path('resources', 'ADAPT.conf') as path:
    ADAPT_CONF_PATH = path.name
with resources.path('resources', 'MC.conf') as path:
    MC_CONF_PATH = path.name
