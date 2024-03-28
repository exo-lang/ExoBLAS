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


CblasUpperValue = CBLAS_UPLO.CblasUpper.value
CblasLowerValue = CBLAS_UPLO.CblasLower.value
CblasLeftValue = CBLAS_SIDE.CblasLeft.value
CblasRightValue = CBLAS_SIDE.CblasRight.value
CblasNoTransValue = CBLAS_TRANSPOSE.CblasNoTrans.value
CblasTransValue = CBLAS_TRANSPOSE.CblasTrans.value

Cblas_suffix = {
    CblasNoTransValue: "n",
    CblasTransValue: "t",
    CblasLeftValue: "l",
    CblasRightValue: "r",
    CblasUpperValue: "u",
    CblasLowerValue: "l",
}

Cblas_params_defaults = {
    "Side": CblasLeftValue,
    "Uplo": CblasUpperValue,
    "TransA": CblasNoTransValue,
    "TransB": CblasNoTransValue,
}
