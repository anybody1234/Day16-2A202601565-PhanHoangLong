"""`budget_policy`'s evidence FLOOR — the real-model failure, reproduced.

`tests/test_runner.py` records the measurement this exists for: on a live
endpoint, gpt-5.6-luna abstained on turn 1 with ZERO tool calls on 4 of 6
runs. Zero tools -> zero evidence -> zero claims -> the abstain floor, and
the whole ladder flattens with nothing in the trace saying why.

The instructor-owned patch (`RunnerConfig.prompt_addendum`) defaults to
OFF and the scored runner overrides `system_prompt` anyway, so a student
cannot reach it. `BudgetPolicy.after_model` is the student-owned lever,
and this suite is the evidence that it works — reproducing the failure
with a scripted model instead of paying an endpoint for the privilege.
"""

from __future__ import annotations

from arena.model import ModelResponse, parse_output, render_final
from arena.tools import Tools
from arena.trace import Trace

from harness.agent import ReActAgent
from harness.layers.budget_policy import BudgetPolicy

from tests.fixtures_briefs import BRIEF_SLA, CORPUS, SEED

ABSTAIN_FINAL = render_final(
    "Tôi không có đủ thông tin.",
    {
        "answer": "Không đủ căn cứ để trả lời câu hỏi này.",
        "claims": [],
        "citations": [],
        "abstain": True,
    },
)


class AbstainsOnTurnOne:
    """The measured failure: a FINAL on turn 1, before any tool call.

    Keeps abstaining until it is shown an observation, then answers — a
    model that gave up early, not one that is broken.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        saw_observation = any(
            message.get("role") == "user" and "doc-" in str(message.get("content", ""))
            for message in messages[2:]
        )
        text = ABSTAIN_FINAL
        if saw_observation:
            text = render_final(
                "Đã có bằng chứng.",
                {
                    "answer": "Đã tìm được tài liệu liên quan.",
                    "claims": [],
                    "citations": [],
                    "abstain": False,
                },
            )
        return ModelResponse(text=text, prompt_tokens=10, completion_tokens=10)


def _run(middleware):
    trace = Trace(run_id="floor", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=False)
    model = AbstainsOnTurnOne()
    agent = ReActAgent(model, tools, trace, middleware=middleware, corpus=CORPUS)
    report = agent.run(BRIEF_SLA)
    return report, tools, trace.to_jsonl()


def test_without_the_floor_the_run_spends_nothing_and_sees_nothing():
    """The baseline this layer exists to beat — not a bug, the status quo."""
    report, tools, _ = _run(None)
    assert tools.calls == 1, tools.calls  # the submit, and nothing else
    assert report["abstain"] is True


def test_the_floor_forces_one_search_before_the_first_conclusion():
    report, tools, jsonl = _run([BudgetPolicy()])

    assert "search" in jsonl, "the forced turn must reach the frozen tool layer"
    assert tools.calls >= 2, tools.calls  # search + submit
    # The run recovered: it reached a conclusion with evidence behind it.
    assert report["abstain"] is False


def test_the_floor_fires_at_most_once():
    """A layer that re-fired every turn would loop until `max_steps`."""
    _, tools, jsonl = _run([BudgetPolicy()])
    assert jsonl.count('"name": "search"') == 1, jsonl.count('"name": "search"')


def test_the_deferred_final_is_kept_rather_than_lost():
    """Deferring may buy a turn; it may never cost the report.

    A model that abstains and then never writes another FINAL must still
    submit the abstention it did write — otherwise the floor would turn a
    low score into an empty one.
    """

    class OnlyEverAbstains(AbstainsOnTurnOne):
        def complete(self, messages):
            return ModelResponse(text=ABSTAIN_FINAL, prompt_tokens=10, completion_tokens=10)

    trace = Trace(run_id="floor-stash", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=False)
    agent = ReActAgent(
        OnlyEverAbstains(), tools, trace, middleware=[BudgetPolicy()], corpus=CORPUS
    )
    report = agent.run(BRIEF_SLA)

    assert report.get("abstain") is True
    assert report.get("answer"), "the deferred payload must come back, not an empty dict"


def test_the_floor_leaves_a_model_that_already_works_alone():
    """It must be inert whenever the model does its own retrieval — which
    is why the mock path measures identically with and without it."""

    class SearchesFirst:
        def __init__(self) -> None:
            self.seen = []

        def complete(self, messages):
            self.seen.append(messages)
            if len(self.seen) == 1:
                from arena.model import render_action

                text = render_action("Tìm trước.", "search", {"query": "SLA", "k": 5})
            else:
                text = ABSTAIN_FINAL
            return ModelResponse(text=text, prompt_tokens=10, completion_tokens=10)

    trace = Trace(run_id="floor-inert", seed=SEED)
    tools = Tools(CORPUS, trace, seed=SEED, flaky=False)
    model = SearchesFirst()
    agent = ReActAgent(
        model, tools, trace, middleware=[BudgetPolicy()], corpus=CORPUS
    )
    agent.run(BRIEF_SLA)

    # Exactly the model's own search plus the submit: the layer added nothing.
    assert tools.calls == 2, tools.calls
    assert parse_output(model.seen and ABSTAIN_FINAL).kind == "final"
