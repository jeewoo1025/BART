import torch
from torch.optim.lr_scheduler import LambdaLR

from .utils import get_logger
logger = get_logger("train")

class LinearWarmupLR(LambdaLR):
    """
    LR Scheduling function which is increase lr on warmup steps and decrease on normal steps
        * LambdaLR : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
        * 초기 learning rate에 lambda 함수에서 나온값을 곱해줘서 learning rate를 계산한다.
    """
    def __init__(
        self, 
        optimizer :torch.optim.Optimizer,
        num_warmup_steps :int,
        max_lr :float,
        last_epoch :int=-1,
        verbose :bool=False,
    ) -> None:
        """
        Args:
            optimizer: torch optimizer
            num_warmup_steps: number of warmup steps
            max_lr : max learning rate
        """
        self.num_warmup_steps = num_warmup_steps
        self.max_lr = max_lr

        super().__init__(optimizer, self._get_lr, last_epoch=last_epoch, verbose=verbose)

    
    def _get_lr(self, current_step :int) -> float:
        if current_step == 0:
            return 0.0

        return self.max_lr*min(current_step**(-0.5), current_step*(self.num_warmup_steps**(-1.5)))