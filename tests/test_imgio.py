"""imgio 단위 테스트: 한글 경로 imread/imwrite 라운드트립 (F1 수정)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bubble_counter.imgio import imread_unicode, imwrite_unicode


def _img() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(12, 20, 3), dtype=np.uint8)


def test_round_trip_ascii_path(tmp_path):
    img = _img()
    path = tmp_path / "plain.png"
    assert imwrite_unicode(path, img) is True
    back = imread_unicode(path)
    assert back is not None and np.array_equal(back, img)


def test_round_trip_korean_path(tmp_path):
    # F1 핵심: 경로에 한글이 있어도 저장·로드가 손실 없이 왕복해야 한다.
    img = _img()
    path = tmp_path / "한글폴더" / "기포_이미지.png"
    path.parent.mkdir(parents=True)
    assert imwrite_unicode(path, img) is True
    back = imread_unicode(path)
    assert back is not None and np.array_equal(back, img)


def test_imread_missing_returns_none(tmp_path):
    # cv2.imread 계약: 없는 파일은 예외가 아니라 None.
    assert imread_unicode(tmp_path / "없음.png") is None


def test_imread_grayscale_flag(tmp_path):
    path = tmp_path / "g.png"
    imwrite_unicode(path, _img())
    gray = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    assert gray is not None and gray.ndim == 2


def test_imwrite_bad_dir_returns_false(tmp_path):
    # 존재하지 않는 폴더로의 저장은 예외 없이 False.
    assert imwrite_unicode(tmp_path / "없는폴더" / "x.png", _img()) is False


def test_imwrite_unsupported_ext_returns_false(tmp_path):
    # 인코더 없는 확장자는 예외 없이 False.
    assert imwrite_unicode(tmp_path / "x.zzz", _img()) is False


def test_imwrite_matches_cv2_imread_for_ascii(tmp_path):
    # imgio로 쓴 PNG를 기존 cv2.imread가 그대로 읽어야 한다(기존 테스트 회귀 안전).
    img = _img()
    path = tmp_path / "compat.png"
    imwrite_unicode(path, img)
    back = cv2.imread(str(path))
    assert back is not None and np.array_equal(back, img)
