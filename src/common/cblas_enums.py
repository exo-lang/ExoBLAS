from enum import Enum

# From netlib `cblas.h`
class CBLAS_TRANSPOSE(Enum):
    CblasNoTrans = 111
    CblasTrans = 112
    CblasConjTrans = 113


class CBLAS_SIDE(Enum):
    CblasLeft = 141
    CblasRight = 142


class CBLAS_UPLO(Enum):
    CblasUpper = 121
    CblasLower = 122


class CBLAS_DIAG(Enum):
    CblasNonUnit = 131
    CblasUnit = 132


CblasUpperValue = CBLAS_UPLO.CblasUpper.value
CblasLowerValue = CBLAS_UPLO.CblasLower.value
CblasLeftValue = CBLAS_SIDE.CblasLeft.value
CblasRightValue = CBLAS_SIDE.CblasRight.value
CblasNoTransValue = CBLAS_TRANSPOSE.CblasNoTrans.value
CblasTransValue = CBLAS_TRANSPOSE.CblasTrans.value
CblasConjTransValue = CBLAS_TRANSPOSE.CblasConjTrans.value
CblasNonUnitValue = CBLAS_DIAG.CblasNonUnit.value
CblasUnitValue = CBLAS_DIAG.CblasUnit.value

Cblas_suffix = {
    CblasNoTransValue: "n",
    CblasTransValue: "t",
    CblasConjTransValue: "t",
    CblasLeftValue: "l",
    CblasRightValue: "r",
    CblasUpperValue: "u",
    CblasLowerValue: "l",
}

TransVals = (CblasNoTransValue, CblasTransValue, CblasConjTransValue)

Cblas_params_values = {
    "Side": (CblasLeftValue, CblasRightValue),
    "Uplo": (CblasUpperValue, CblasLowerValue),
    "TransA": TransVals,
    "TransB": TransVals,
    "Trans": TransVals,
}
