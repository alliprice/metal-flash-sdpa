"""Shared fixtures and configuration for metal-flash-sdpa tests."""
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    if not torch.backends.mps.is_available():
        pytest.exit("MPS device not available — tests require Apple Silicon", returncode=0)


@pytest.fixture(autouse=True)
def _mfa_cleanup():
    """Ensure MFA is disabled and dispatch counter reset after every test."""
    import metal_flash_sdpa
    metal_flash_sdpa.reset_dispatch_count()
    # Reset mask cache to prevent cross-test pollution
    metal_flash_sdpa._last_mask_ptr = None
    metal_flash_sdpa._last_mask_trivial = False
    yield
    metal_flash_sdpa.disable()
