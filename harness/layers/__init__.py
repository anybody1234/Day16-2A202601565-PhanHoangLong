"""Phần xét đoán dùng chung cho năm lớp.

VÌ SAO CÓ FILE NÀY: `critic` và `citation_checker` phải trả lời CÙNG MỘT
câu hỏi — "câu này có phải trích dẫn nguyên văn một DÒNG của tài liệu
không?" — và nếu hai lớp trả lời khác nhau thì lớp này giữ đúng cái lớp
kia vừa vứt đi. Viết một lần, dùng chung.

NGUYÊN TẮC: KHÔNG viết lại phép thử của bộ chấm, mà IMPORT chính nó từ
`arena/scorer.py`. Năm file trong `arena/` bị khoá bằng MD5 (xem
`scripts/verify.py`), nên `_norm` / `_supports` / `_norm_lines` không thể
đổi dưới chân ta; còn một bản chép tay thì sẽ lệch âm thầm đúng vào lúc
không ai kiểm tra. Lệch một ký tự ở đây = claim bị chấm HALLUCINATED =
mất trọn 15 điểm honesty.

BA CÁI BẪY MÀ CÁC HÀM DƯỚI ĐÂY CHẶN SẴN:

  * `text in doc.body` là SAI. Bộ chấm chỉ nhận trích dẫn khớp một DÒNG
    (`_supports`), nên một câu vắt qua hai dòng lọt qua phép thử `in
    body` nhưng vẫn bị chấm HALLUCINATED.
  * `ctx.saw(text)` trần là SAI. `_supports` bỏ qua mọi chuỗi ngắn hơn
    `MIN_SUPPORT_CHARS` (12) — một chuỗi 5 ký tự vẫn `saw() == True`
    trong khi bộ chấm không cho nó điểm nào.
  * "tài liệu đã đọc" không phải "tài liệu có trong corpus". Trích một
    tài liệu lượt chạy chưa từng chạm bị chấm UNRETRIEVED (phạt 0.75).
"""

from __future__ import annotations

import re

from arena.scorer import (  # phép thử của chính bộ chấm, không phải bản chép
    MAX_CLAIM_CHARS,
    MAX_CLAIMS_PER_DOC,
    MAX_SCORED_CLAIMS,
    _norm,
    _norm_lines,
    _normalised_bodies,
    _supports,
)

__all__ = [
    "MAX_CLAIM_CHARS",
    "MAX_CLAIMS_PER_DOC",
    "MAX_SCORED_CLAIMS",
    "Evidence",
    "claim_doc_id",
    "claim_text",
    "norm_text",
    "quotes_a_line",
    "safe_to_nudge",
    "trim_to_norm_limit",
]

#: `search` trả về JSON có trường "doc_id"; `fetch_doc` chỉ trả về body
#: trần. Không một tài liệu nào trong kho nhắc tới mã của tài liệu khác
#: (đã kiểm: 0/120), nên regex này chỉ bắt đúng mã đã về từ một lần
#: search — không có dương tính giả.
_DOC_ID_RE = re.compile(r"doc-\d{4}")


def norm_text(text) -> str:
    """Dạng chuẩn hoá mà MỌI phép so sánh của bộ chấm chạy trên đó."""
    return _norm(text)


def safe_to_nudge(messages) -> bool:
    """Đã qua lượt assistant đầu tiên chưa? Nếu chưa, ĐỪNG chèn message.

    `arena.model._first_user_content` lấy message user CUỐI CÙNG TRƯỚC
    lượt assistant đầu tiên làm câu hỏi của brief, và chỉ bỏ qua những
    message mang `FINALIZE_SENTINEL`. Một câu nhắc trơn chèn vào phần mở
    đầu sẽ TRỞ THÀNH câu hỏi cho cả lượt chạy: mọi brief truy xuất cùng
    một mớ tài liệu vô can và không có một dòng lỗi nào.

    Sau lượt assistant đầu tiên, vùng nguy hiểm đó đã đóng — nhắc thoải
    mái. Đây là lý do mọi hook `before_model` trong package này gọi hàm
    này trước khi thêm bất cứ thứ gì.
    """
    return any(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in messages
    )


def quotes_a_line(body: str, text: str) -> bool:
    """`text` có phải trích dẫn nguyên văn MỘT DÒNG của `body` không?

    Bản độc lập, dùng khi trong tay chỉ có một chuỗi body. Trong
    `after_agent` hãy dùng `Evidence.supports` — nó đọc bảng dòng đã
    chuẩn hoá sẵn có cache thay vì chuẩn hoá lại mỗi lần gọi.
    """
    return _supports(_norm_lines(body), _norm(text))


def claim_text(claim) -> str:
    """`claim["text"]` nếu nó là chuỗi, ngược lại chuỗi rỗng."""
    if not isinstance(claim, dict):
        return ""
    value = claim.get("text")
    return value if isinstance(value, str) else ""


def claim_doc_id(claim) -> str:
    if not isinstance(claim, dict):
        return ""
    value = claim.get("doc_id")
    return value if isinstance(value, str) else ""


def trim_to_norm_limit(text: str, limit: int = MAX_CLAIM_CHARS) -> str:
    """Cắt `text` sao cho `len(norm_text(...)) <= limit`.

    CẮT BỚT là phép sửa hợp lệ duy nhất trên `claim["text"]`: một khúc
    đầu của một dòng vẫn là trích dẫn nguyên văn của dòng đó, và vẫn là
    chữ mô hình đã viết. Trần được đo trên bản ĐÃ chuẩn hoá
    (`scorer._classify_claims`), nên cắt theo `len(text)` trần là không
    đủ khi câu có nhiều khoảng trắng liên tiếp.
    """
    if not isinstance(text, str) or len(_norm(text)) <= limit:
        return text
    # `len(_norm(text[:n]))` không giảm khi `n` tăng -> chặt nhị phân tìm
    # khúc đầu dài nhất còn lọt trần.
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(_norm(text[:middle])) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low]


class Evidence:
    """Một lần đọc bằng chứng của lượt chạy, dùng lại cho mọi claim.

    Dựng MỘT LẦN ở đầu `after_agent`: chuẩn hoá `ctx.observed_text` tốn
    kém và cả `critic` lẫn `citation_checker` đều cần đúng nó.
    """

    def __init__(self, ctx) -> None:
        corpus = getattr(ctx, "corpus", None)
        self.corpus = corpus
        # {doc_id: tuple các dòng đã chuẩn hoá}, có cache theo Corpus.
        self.bodies = _normalised_bodies(corpus) if corpus is not None else {}
        self.observed = _norm(getattr(ctx, "observed_text", "") or "")
        self.doc_ids = self._observed_doc_ids()

    def _observed_doc_ids(self) -> set:
        """Những tài liệu lượt chạy này thật sự đã nhìn thấy.

        Xấp xỉ THẬN TRỌNG của `run.retrieved` bên bộ chấm — bộ chấm còn
        chạy lại từng câu search từ trace nên tập của nó rộng hơn. Thiếu
        một mã ở đây chỉ làm ta bỏ lỡ một lần gắn lại; thừa một mã thì
        tạo ra UNRETRIEVED. Nên thà thiếu.

        Hai nguồn, vì hai công cụ trả về hai thứ khác nhau:
          * `search` -> JSON có "doc_id"  -> bắt bằng regex.
          * `fetch_doc` -> body trần, KHÔNG có mã -> nhận ra tài liệu qua
            chính body của nó (bản bị cắt hay bị nhiễu sẽ không khớp, và
            đúng là không nên khớp).
        """
        seen = set(_DOC_ID_RE.findall(self.observed))
        for doc_id, lines in self.bodies.items():
            if doc_id not in seen and lines and " ".join(lines) in self.observed:
                seen.add(doc_id)
        return seen

    def saw(self, text: str) -> bool:
        """Câu này có nằm nguyên văn trong bằng chứng agent đã đọc không?

        Chuẩn hoá cả hai vế — `ctx.saw` so sánh chuỗi thô nên trượt ngay
        khi quan sát và câu trích khác nhau đúng một khoảng trắng.
        """
        normalised = _norm(text)
        return bool(normalised) and normalised in self.observed

    def supports(self, doc_id: str, text: str) -> bool:
        """Tài liệu `doc_id` có nói câu này, DƯỚI DẠNG TRÍCH DẪN, không?"""
        return _supports(self.bodies.get(doc_id, ()), _norm(text))

    def source_of(self, text: str):
        """Tài liệu ĐÃ QUAN SÁT đầu tiên trích dẫn được câu này, hoặc None.

        Chỉ xét tài liệu trong `doc_ids`: gắn claim vào một tài liệu đúng
        nội dung nhưng lượt chạy chưa đọc thì đổi HALLUCINATED (1.0) lấy
        UNRETRIEVED (0.75) — rẻ hơn một chút, nhưng vẫn là bịa.
        """
        normalised = _norm(text)
        if not normalised or self.corpus is None:
            return None
        for doc in self.corpus.docs:
            if doc.doc_id in self.doc_ids and _supports(
                self.bodies.get(doc.doc_id, ()), normalised
            ):
                return doc.doc_id
        return None
