from rlpyt.utils.logging import logger
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
import numpy as np


def VaR(values, alpha):
    index = int(np.round(len(values) * alpha))
    return sorted(values)[index]


def LeftCVaR(values, alpha):
    index = int(np.round(len(values) * alpha))
    return np.mean(sorted(values)[:index + 1])


def RightCVaR(values, alpha):
    index = int(np.round(len(values) * alpha))
    return np.mean(sorted(values)[index:])


def SD_Down(values):
    values = np.asarray(values)
    return np.sqrt(np.power((values - values.mean()).clip(None, 0), 2).mean())


def SD_Up(values):
    values = np.asarray(values)
    return np.sqrt(np.power((values - values.mean()).clip(0, None), 2).mean())


class RiskRLMixin(object):

    def _log_infos(self, traj_infos=None):
        super()._log_infos(traj_infos)
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if 'Return' in k:
                    values = [info[k] for info in traj_infos]
                    logger.record_tabular(k + 'SD(Down)', SD_Up(values))
                    logger.record_tabular(k + 'SD(Up)', SD_Down(values))
                    for alpha in [0.25, 0.75]:
                        logger.record_tabular(k + 'VaR' + str(alpha), VaR(values, alpha))
                        # logger.record_tabular(k + 'LeftCVaR' + str(alpha), LeftCVaR(values, alpha))
                        # logger.record_tabular(k + 'RightCVaR' + str(alpha), RightCVaR(values, alpha))


class RiskMinibatchRl(RiskRLMixin, MinibatchRl):
    pass


class RiskMinibatchRlEval(RiskRLMixin, MinibatchRlEval):
    pass
