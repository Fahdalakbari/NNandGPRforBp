import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# Hyperparameters
# ==========================================================

acti1 = nn.Tanh
acti2 = nn.Tanh
acti3 = nn.Tanh

neu1 = 330
neu2 = 330
neu3 = 330


# ==========================================================
# PINN Model
# ==========================================================

class PINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(12, neu1),
            acti1(),

            nn.Linear(neu1, neu2),
            acti2(),

            nn.Linear(neu2, neu3),
            acti3(),

            nn.Linear(neu3, 1)

        )

    def forward(self, x):

        return F.softplus(self.net(x))
