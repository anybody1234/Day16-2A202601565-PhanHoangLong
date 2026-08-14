"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from arena.model import (
    FINALIZE_SENTINEL,
    ModelResponse,
    parse_output,
    render_action,
)
from arena.tools import ToolResult

from harness.agent import (
    _action_under_final,
    _canonicalise,
    _is_report_payload,
)
from harness.middleware import Middleware

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

#: Số tài liệu xin về ở lượt search bắt buộc. Giữ đúng mặc định của giao
#: thức — `arena/runner.py` kẹp k về <= 10 VÀ gắn cờ mọi đường đi vòng.
FLOOR_SEARCH_K = 5

FLOOR_THOUGHT = (
    "Tôi chưa đọc tài liệu nào, nên chưa thể kết luận. Tôi tìm bằng chứng trước."
)

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(
        self,
        reserve: int = DEFAULT_RESERVE,
        force_evidence: bool = True,
    ) -> None:
        self.reserve = max(0, int(reserve))
        self.force_evidence = bool(force_evidence)

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        if not self._spent(ctx):
            return messages
        # `messages + [...]` chứ không `append`: nhắc trong đúng lượt này.
        return messages + [{"role": "user", "content": NUDGE}]

    #: Nơi cất FINAL bị hoãn ở `after_model`, để `after_agent` còn nộp
    #: lại được nếu lượt chạy không bao giờ chốt một FINAL nào khác.
    _STASH = "budget_policy_deferred_final"

    def after_model(self, ctx, response):
        """SÀN của ngân sách: cấm kết luận khi chưa tiêu một lượt nào.

        Cùng một câu hỏi với hai hook kia ("agent được phép dừng khi
        nào?"), chỉ là đầu còn lại: `before_model`/`wrap_tool_call` chặn
        TIÊU QUÁ, hook này chặn KHÔNG TIÊU GÌ.

        VÌ SAO CẦN: `tests/test_runner.py` đo trên endpoint thật —
        gpt-5.6-luna abstain ngay lượt 1 với KHÔNG một lượt gọi công cụ
        nào ở 4/6 lần chạy. Không tool call -> không bằng chứng -> không
        claim -> sàn abstain, và cả thang điểm phẳng ra. Bản vá chính
        thức (`RunnerConfig.prompt_addendum`) MẶC ĐỊNH TẮT và do người
        chấm bật, không phải sinh viên — nên chỗ vá của sinh viên chỉ có
        thể là một hook.

        HỢP LỆ: `after_model` đổi thứ AGENT LÀM, không đổi thứ bộ chấm
        tin là chữ mô hình — `_call_model` đã đóng dấu `model_call` với
        output THÔ trước khi hook này chạy (xem `harness/agent.py`).

        Bắn ĐÚNG MỘT LẦN và chỉ khi cả bốn điều kiện dưới đây cùng đúng,
        nên trên đường mock (mô hình luôn mở đầu bằng ACTION) nó không
        bao giờ chạy — đã đo: 5 seed, điểm không đổi.
        """
        if not self.force_evidence or ctx.state.get(self._STASH) is not None:
            return response
        if getattr(ctx.tools, "calls", 0) or self._spent(ctx):
            return response
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            return response

        # Đọc lượt này ĐÚNG như agent sẽ đọc, nếu không hai bên bất đồng
        # về chuyện "lượt này có kết thúc run không".
        parsed = parse_output(_canonicalise(text))
        if parsed.kind != "final" or not _is_report_payload(parsed.final):
            return response
        if _action_under_final(text) is not None:
            return response  # model đã tự viết ACTION bên dưới: để nó chạy
        question = (ctx.question or "").strip()
        if not question:
            return response

        # Giữ lại payload bị hoãn: nếu lượt chạy không bao giờ chốt FINAL
        # nào khác thì `after_agent` nộp lại nó. Hoãn chỉ được phép mua
        # thêm một lượt, không được phép làm mất cả báo cáo.
        ctx.state[self._STASH] = parsed.final
        return ModelResponse(
            text=render_action(
                FLOOR_THOUGHT, "search", {"query": question, "k": FLOOR_SEARCH_K}
            ),
            prompt_tokens=getattr(response, "prompt_tokens", 0),
            completion_tokens=getattr(response, "completion_tokens", 0),
        )

    def after_agent(self, ctx, report):
        """Nộp lại FINAL đã hoãn nếu agent kết thúc mà không có cái nào."""
        stashed = ctx.state.get(self._STASH)
        if isinstance(stashed, dict) and not (
            isinstance(report, dict) and report
        ):
            return dict(stashed)
        return report

    def wrap_tool_call(self, ctx, call, name, args):
        if not self._spent(ctx):
            return call(name, args)
        # Không gọi `call(...)` CHÍNH LÀ cách chặn. Không raise: agent
        # phải sống sót để còn chốt FINAL.
        ctx.state["budget_policy_refused"] = (
            int(ctx.state.get("budget_policy_refused", 0)) + 1
        )
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"ngân sách công cụ đã hết ({ctx.tools.calls}/{ctx.max_tool_calls}); "
                "hãy trả lời ngay bằng bằng chứng đang có"
            ),
        )
