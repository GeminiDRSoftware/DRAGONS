# GMOS read modes. Dict for EEV previously defined in GMOS_Descriptors.py
# for descriptor read_mode().
# 'default' applies for both EEV and the super old e2v CCDs.
#
# It is unclear whether there is an 'Engineering' read mode for 
# Hamamatsu CCDs. This mode is not defined in the requirements document,
# GMOS-S Hamamatsu CCD Upgrades, v2.0 - 15 December 2014 (M Simpson).
# The mode is left in here for JIC purposes as the final possible combination
# [ gain_setting, read_speed_setting ].

# 04-06-2015, kra
read_mode_map = {
    "default": { 
        "Normal": ["low", "slow"],
        "Bright": ["hgh", "fast"],
        "Acquisition": ["low", "fast"],
        "Engineering": ["high", "slow"],
    },
    "Hamamatsu": { 
        "Normal": ["low", "slow"],
        "Bright": ["high", "fast"],
        "Acquisition": ["low", "fast"],
        "Engineering": ["high", "slow"],
    }
}
