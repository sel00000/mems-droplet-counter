"""한글(비-ASCII) 경로에서도 동작하는 이미지 IO — cv2.imread/imwrite 드롭인.

Windows의 cv2.imgcodecs는 ANSI fopen을 사용해 한글 경로 파일을 열지 못한다
(리포트 F1). 파이썬이 바이트를 직접 읽고/쓰고(np.fromfile / ndarray.tofile은
유니코드 경로를 지원), 인코딩·디코딩만 OpenCV에 맡겨 이 문제를 피한다.

프로젝트의 모든 이미지 파일 IO는 이 모듈을 경유한다(cv2.imread/imwrite 직접
호출 금지). 영상 IO(VideoCapture/VideoWriter)는 별개 경로라 대상이 아니다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: "str | Path", flags: int = cv2.IMREAD_COLOR) -> "np.ndarray | None":
    """cv2.imread 드롭인. 파일 없음/디코드 실패 시 None(예외 없음)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: "str | Path", img: "np.ndarray",
                    ext: "str | None" = None) -> bool:
    """cv2.imwrite 드롭인. 성공 True / 실패 False(예외 없음).

    확장자는 ext 인자 → 경로 suffix → ".png" 순으로 결정한다. imencode가 지원하지
    않는 확장자거나 대상 폴더가 없으면(둘 다 실 사용에서 발생) False를 반환한다.
    """
    p = Path(path)
    suffix = ext if ext is not None else (p.suffix or ".png")
    try:
        ok, buf = cv2.imencode(suffix, img)
    except cv2.error:
        return False
    if not ok:
        return False
    try:
        buf.tofile(str(p))
    except OSError:
        return False
    return True
