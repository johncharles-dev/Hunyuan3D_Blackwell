import torch
from typing import Literal, Optional, Tuple


class VRAMManager:
    """
    Auto-detect GPU VRAM and set behavior tier for Hunyuan3D pipeline.

    Tiers:
        high   (>=24GB): float32, all models on GPU, no swapping
        medium (12-23GB): float16, model swapping between shape/texture stages
        low    (8-11GB):  float16, aggressive swapping, reduced texture views, CPU rembg
    """

    TIER_HIGH = 'high'
    TIER_MEDIUM = 'medium'
    TIER_LOW = 'low'

    def __init__(
        self,
        precision: Literal['auto', 'full', 'half'] = 'auto',
        vram_tier: Literal['auto', 'high', 'medium', 'low'] = 'auto',
        device_id: int = 0,
    ):
        self._device_id = device_id
        self._vram_mb = self._detect_vram()
        self._tier = self._resolve_tier(vram_tier)
        self._precision = precision
        self._dtype = self._resolve_dtype(precision)

    def _detect_vram(self) -> int:
        if not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(self._device_id)
        return props.total_memory // (1024 * 1024)

    def _resolve_tier(self, vram_tier: str) -> str:
        if vram_tier != 'auto':
            return vram_tier
        if self._vram_mb >= 24000:
            return self.TIER_HIGH
        elif self._vram_mb >= 12000:
            return self.TIER_MEDIUM
        else:
            return self.TIER_LOW

    def _resolve_dtype(self, precision: str) -> torch.dtype:
        if precision == 'full':
            return torch.float32
        elif precision == 'half':
            return torch.float16
        else:  # auto
            if self._tier == self.TIER_HIGH:
                return torch.float32
            else:
                return torch.float16

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def vram_mb(self) -> int:
        return self._vram_mb

    @property
    def use_model_swapping(self) -> bool:
        return self._tier != self.TIER_HIGH

    @property
    def rembg_provider(self) -> str:
        if self._tier == self.TIER_LOW:
            return 'CPUExecutionProvider'
        return 'CUDAExecutionProvider'

    @property
    def texture_config(self) -> Tuple[int, int]:
        """Return (max_num_view, resolution) based on VRAM tier."""
        if self._tier == self.TIER_HIGH:
            return (9, 1024)
        elif self._tier == self.TIER_MEDIUM:
            return (8, 768)
        else:
            return (6, 512)

    def __repr__(self) -> str:
        views, res = self.texture_config
        return (
            f"VRAMManager(tier={self._tier}, dtype={self._dtype}, "
            f"vram={self._vram_mb}MB, swapping={self.use_model_swapping}, "
            f"texture={views}views@{res}px)"
        )


# Singleton instance — initialized by entry points
_global_vram_manager: Optional[VRAMManager] = None


def get_vram_manager() -> VRAMManager:
    global _global_vram_manager
    if _global_vram_manager is None:
        _global_vram_manager = VRAMManager()
    return _global_vram_manager


def init_vram_manager(
    precision: Literal['auto', 'full', 'half'] = 'auto',
    vram_tier: Literal['auto', 'high', 'medium', 'low'] = 'auto',
) -> VRAMManager:
    global _global_vram_manager
    _global_vram_manager = VRAMManager(precision=precision, vram_tier=vram_tier)
    print(f"[VRAMManager] {_global_vram_manager}")
    return _global_vram_manager
