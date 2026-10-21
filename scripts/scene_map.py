"""
이미지 → 장면 토큰 매핑 (평가 데이터 전부로 통일, 사전 등록 4fddeb4).

DriveLM 검증 JSON은 scene → key_frames → image_paths 구조다. 이미지별 결과에 장면 토큰을 붙여야
**장면 군집 부트스트랩**으로 신뢰구간을 낼 수 있다(799장 = 149장면, 한 장면 안의 이미지는 서로 독립이 아니다).

    from scene_map import image_to_scene
    scene = image_to_scene()["n008-...__CAM_FRONT__1537295990612404.jpg"]
"""
import json
from functools import lru_cache
from pathlib import Path

DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"


@lru_cache(maxsize=None)
def image_to_scene(drivelm_json: str = DRIVELM_VAL) -> dict:
    """{이미지 basename: scene_token} — 모든 카메라."""
    d = json.load(open(drivelm_json))
    out = {}
    for scene, sd in d.items():
        for _ft, fd in sd["key_frames"].items():
            for _cam, rel in fd.get("image_paths", {}).items():
                out[Path(rel).name] = scene
    return out


@lru_cache(maxsize=None)
def token_to_scene(drivelm_json: str = DRIVELM_VAL) -> dict:
    """{frame(sample) token: scene_token} — NuScenes-QA 문항별 결과용."""
    d = json.load(open(drivelm_json))
    return {ft: scene for scene, sd in d.items() for ft in sd["key_frames"]}
