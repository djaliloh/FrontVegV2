import pytest

from frontveg.config import FrontVegConfig
from frontveg.resources import ENV_SAM3_CKPT, resolve_paths


def test_config_validation():
    with pytest.raises(ValueError):
        FrontVegConfig(overlap=1.0)
    with pytest.raises(ValueError):
        FrontVegConfig(peak_dist=0)


def test_config_from_mapping_ignores_unknown_and_none():
    cfg = FrontVegConfig.from_mapping({"sigma": 2.0, "prompt": None, "input": "x"})
    assert cfg.sigma == 2.0 and cfg.prompt == FrontVegConfig().prompt


def test_resolve_paths_from_root_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_SAM3_CKPT, raising=False)
    paths = resolve_paths(tmp_path)
    assert paths.sam3_ckpt == tmp_path / "checkpoints" / "sam3_ckpts" / "sam3.pt"
    assert paths.depth_ckpt("vitl").name == "depth_anything_v2_vitl.pth"

    monkeypatch.setenv(ENV_SAM3_CKPT, str(tmp_path / "custom.pt"))
    assert resolve_paths(tmp_path).sam3_ckpt == tmp_path / "custom.pt"


def test_missing_lists_every_absent_resource(tmp_path):
    assert len(resolve_paths(tmp_path).missing("vitl", sam3=False)) == 2

    (tmp_path / "external" / "Depth-Anything-V2" / "depth_anything_v2").mkdir(parents=True)
    ckpt_dir = tmp_path / "checkpoints" / "depthanything_ckpts"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "depth_anything_v2_vitl.pth").touch()
    assert resolve_paths(tmp_path).missing("vitl", sam3=False) == []
