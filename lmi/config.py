from dataclasses import dataclass
import numpy as np
C0 = 299792458.0
DT = 1e-11
NT = 1024
T0 = 6e-10
FP = 2200000000.0
BAND = (1000000000.0, 4000000000.0)
FRAME_RATE = 61.0
_P = 0.075
TX_POS = np.array([[-_P, -_P / 2], [_P, _P / 2]])
RX_POS = np.array([[0.0, -_P / 2], [_P, -_P / 2], [-_P, _P / 2], [0.0, _P / 2]])
PAIRS = np.array([(t, r) for t in range(len(TX_POS)) for r in range(len(RX_POS))])
N_PAIRS = len(PAIRS)
FS_VIB = 4000.0
MMWAVE_LAMBDA = C0 / 62000000000.0
CHIRP_F = (100.0, 800.0)
CHIRP_DUR = 0.2
QUIET_DUR = 0.05

def topp(theta: float) -> float:
    return 3.03 + 9.3 * theta + 146.0 * theta ** 2 - 76.7 * theta ** 3

@dataclass(frozen=True)
class Soil:
    name: str
    eps: float
    alpha: float
    acoustic_damping: float = 1.0
    alpha_exp: float = 0.0

    @property
    def n(self) -> float:
        return float(np.sqrt(self.eps))

    @property
    def v(self) -> float:
        return C0 / self.n
DRY_SAND = Soil('dry sand', 3.2, 0.5, 1.0)
WET_CLAY = Soil('wet clay', round(topp(0.25), 2), 12.0, 1.6)
MEDIUM_LOAM = Soil('loam', round(topp(0.15), 2), 5.0, 1.3)

@dataclass(frozen=True)
class Band:
    fp: float
    lo: float
    hi: float
HIGH_BAND = Band(FP, BAND[0], BAND[1])
LOW_BAND = Band(1200000000.0, 700000000.0, 2500000000.0)
F_REF_LOSS = 2500000000.0
