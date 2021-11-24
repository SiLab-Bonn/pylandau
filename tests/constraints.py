''' Constraints that should be followed to ensure stable and proper results '''

# Value input x
MIN_X = - 1e9
MAX_X = 1e9

# pylandau.get_langau_pdf
# pylandau.langau_pdf
LANDAU_PDF_MIN_MU = -1e9
LANDAU_PDF_MAX_MU = 1e9

# pylandau.landau
LANDAU_MIN_MPV = -1e9
LANDAU_MAX_MPV = 1e9
LANDAU_MIN_ETA = 1e-3
LANDAU_MAX_ETA = 1e9
LANDAU_MIN_A = 0.
LANDAU_MAX_A = 1e9

# pylandau.langau
LANGAU_MIN_MPV = -1e9
LANGAU_MAX_MPV = 1e9
LANGAU_MIN_ETA = 1e-1
LANGAU_MAX_ETA = 1e9
LANGAU_MIN_SIGMA = 1e-1
LANGAU_MAX_SIGMA = 1e9
LANGAU_MIN_A = 0.
LANGAU_MAX_A = 1e9
LANGAU_MAX_ETA_SIGMA = 1e9  # sigma * eta
