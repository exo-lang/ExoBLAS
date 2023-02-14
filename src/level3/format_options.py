"""
Format options for BLAS 3
"""

# UPLO
class ExoBlasUplo:
    def __init__(self):
        return

class ExoBlasUpper(ExoBlasUplo):
    def __init__(self):
        return    

class ExoBlasLower(ExoBlasUplo):
    def __init__(self):
        return

# TRANSPOSE
class ExoBlasT:
    def __init__(self):
        return

class ExoBlasTranspose(ExoBlasT):
    def __init__(self):
        return

class ExoBlasNoTranspose(ExoBlasT):
    def __init__(self):
        return

# DIAGONAL
class ExoBlasDiag:
    def __init__(self):
        return

class ExoBlasUnitDiag(ExoBlasDiag):
    def __init__(self):
        return

class ExoBlasNoUnitDiag(ExoBlasDiag):
    def __init__(self):
        return

# TODO: SIDE