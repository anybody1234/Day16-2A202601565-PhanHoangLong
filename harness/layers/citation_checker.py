"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.layers import Evidence, claim_doc_id, claim_text, safe_to_nudge
from harness.middleware import Middleware


#: Kỷ luật trích dẫn, nhắc lại mỗi lượt sau lượt assistant đầu tiên.
#:
#: ĐO ĐƯỢC trên endpoint thật (gpt-5.6-luna, pub-01): mô hình trích
#: "Thời gian giao hàng cam kết hiện hành: nội thành 2 ngày làm việc;
#: liên tỉnh 5 ngày làm việc." — NỬA ĐẦU của một dòng. Claim đó được
#: chấm SUPPORTED, phạt 0.00... mà recall vẫn 0.00, vì `_covers` đòi
#: >= `FACT_TERM_RATIO` (0.6) số từ khoá của dữ kiện: nửa dòng chỉ phủ
#: 5/11 = 0.45. Chép TRỌN dòng thì phủ 11/11. Chênh lệch: 40.15 -> 100.
#:
#: Không mang `FINALIZE_SENTINEL`: câu này nhắc trích cho ĐÚNG, không
#: bảo dừng.
QUOTE_RULE = (
    "QUY TẮC TRÍCH DẪN (bắt buộc, quyết định điểm): mỗi claim.text phải là "
    "TRỌN MỘT DÒNG của tài liệu, chép y nguyên từ KÝ TỰ ĐẦU TIÊN của dòng "
    "cho tới KÝ TỰ CUỐI CÙNG trước khi xuống dòng.\n"
    "MỘT DÒNG THƯỜNG CHỨA NHIỀU CÂU. Phải chép HẾT các câu đó, không được "
    "dừng ở dấu chấm đầu tiên. Ví dụ, với dòng:\n"
    "    'A: x 2 ngày; y 5 ngày. Mọi B phải dựa trên C.'\n"
    "thì claim.text ĐÚNG là cả dòng đó, còn chép mỗi "
    "'A: x 2 ngày; y 5 ngày.' là SAI và bị chấm 0 điểm.\n"
    "Không tóm tắt, không viết lại, không bỏ vế sau dấu chấm phẩy hay dấu "
    "chấm; giữ nguyên mọi con số, tên riêng và dấu câu."
)


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def __init__(self, remind_quoting: bool = True) -> None:
        self.remind_quoting = bool(remind_quoting)

    def before_model(self, ctx, messages):
        """Nhắc kỷ luật trích dẫn — SAU lượt assistant đầu tiên, luôn luôn.

        `arena.model._first_user_content` đọc message user CUỐI CÙNG
        TRƯỚC lượt assistant đầu tiên làm câu hỏi của brief, và chỉ bỏ
        qua những message mang `FINALIZE_SENTINEL`. Chèn một câu nhắc
        trơn vào phần mở đầu là biến chính câu nhắc thành câu hỏi cho cả
        lượt chạy. Chờ tới khi đã có một lượt assistant thì vùng nguy
        hiểm đó đã đóng lại, nên câu nhắc không thể bị nhầm là brief.
        """
        if not self.remind_quoting or not safe_to_nudge(messages):
            return messages
        # `messages + [...]`, không `append`: nhắc trong đúng lượt này.
        return messages + [{"role": "user", "content": QUOTE_RULE}]

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        evidence = Evidence(ctx)
        rewired = 0

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim_text(claim)
            doc_id = claim_doc_id(claim)

            # Trích dẫn đã đúng: tài liệu được nêu vừa nói câu này dưới
            # dạng trích dẫn, vừa nằm trong số tài liệu lượt chạy đã đọc.
            if doc_id in evidence.doc_ids and evidence.supports(doc_id, text):
                continue

            source = evidence.source_of(text)
            if source is None:
                # Không tài liệu nào đã đọc nói câu này -> BỊA, việc của
                # `critic`. Đừng gán bừa một doc_id: đổi HALLUCINATED lấy
                # MISATTRIBUTED chỉ là giấu lỗi đi cho đẹp.
                continue
            if source != doc_id:
                claim["doc_id"] = source
                rewired += 1

        if rewired:
            ctx.state["citation_checker_rewired"] = rewired
        report["citations"] = sorted(
            {
                claim_doc_id(claim)
                for claim in claims
                if isinstance(claim, dict) and claim_doc_id(claim)
            }
        )
        return report
