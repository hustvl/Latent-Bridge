from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class Omega35AlignmentLossCfg:
    mse_weight: float = 1.0
    cosine_weight: float = 1.0
    temporal_delta_weight: float = 0.5
    temporal_relation_weight: float = 0.1


class Omega35AlignmentLoss(nn.Module):
    def __init__(self, cfg: Omega35AlignmentLossCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Omega35AlignmentLossCfg()
        for name, value in vars(self.cfg).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def temporal_relation(student: Tensor, teacher: Tensor) -> Tensor:
        student_frames = F.normalize(student.float().mean(dim=2), dim=-1)
        teacher_frames = F.normalize(teacher.float().mean(dim=2), dim=-1)
        student_relation = student_frames @ student_frames.transpose(1, 2)
        teacher_relation = teacher_frames @ teacher_frames.transpose(1, 2)
        return F.mse_loss(student_relation, teacher_relation)

    def forward(self, student: Tensor, teacher: Tensor) -> dict[str, Tensor]:
        if student.ndim != 4 or student.shape != teacher.shape:
            raise ValueError(
                "Expected matching [B,T,N,C] tokens, got "
                f"student={tuple(student.shape)} teacher={tuple(teacher.shape)}"
            )
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise ValueError("Alignment tokens contain NaN or Inf")

        student_float = student.float()
        teacher_float = teacher.detach().float()
        mse = F.mse_loss(student_float, teacher_float)
        cosine = 1.0 - F.cosine_similarity(
            student_float,
            teacher_float,
            dim=-1,
        ).mean()
        if student.shape[1] > 1:
            temporal_delta = F.smooth_l1_loss(
                student_float[:, 1:] - student_float[:, :-1],
                teacher_float[:, 1:] - teacher_float[:, :-1],
            )
            temporal_relation = self.temporal_relation(
                student_float,
                teacher_float,
            )
        else:
            temporal_delta = mse.new_zeros(())
            temporal_relation = mse.new_zeros(())

        total = self.cfg.mse_weight * mse
        total = total + self.cfg.cosine_weight * cosine
        total = total + self.cfg.temporal_delta_weight * temporal_delta
        total = total + self.cfg.temporal_relation_weight * temporal_relation
        return {
            "loss": total,
            "mse": mse,
            "cosine": cosine,
            "temporal_delta": temporal_delta,
            "temporal_relation": temporal_relation,
            "student_mean": student_float.mean(),
            "student_std": student_float.std(unbiased=False),
            "teacher_mean": teacher_float.mean(),
            "teacher_std": teacher_float.std(unbiased=False),
        }
