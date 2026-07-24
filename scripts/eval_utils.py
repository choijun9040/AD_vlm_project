"""
평가 스크립트 공용 유틸리티.

NuScenes-QA 정답 정규화 규칙을 한 곳에 모아, 우리 student 모델 평가
(eval_nuscenesqa_val.py)와 SOTA 비교 대상(EM-VLM4AD 등) 평가에 동일하게
적용한다 — 정규화 규칙이 다르면 exact-match 점수가 모델 간에 공정하게
비교되지 않기 때문.
"""

import re

_ARTICLE_PATTERN = re.compile(r"^(a|an|the)\s+")


def normalize_answer(text: str) -> str:
    """대소문자/구두점/관사 차이를 흡수한 뒤 exact match 비교에 사용.

    예: "Yes." -> "yes", "A car" -> "car", "  3  " -> "3"
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)   # 구두점 제거 (마침표 등)
    text = text.strip()
    text = _ARTICLE_PATTERN.sub("", text)  # 선행 관사 제거
    return text.strip()
