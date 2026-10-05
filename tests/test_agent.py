from app.agent import (SEARCH_TOOL, Agent, RetrievalTool, RuleBasedPlanner,
                       SingleShotPlanner, ToolArgs)


def test_tool_schema_is_wellformed():
    assert SEARCH_TOOL["description"].strip()
    props = SEARCH_TOOL["input_schema"]["properties"]
    assert "query" in SEARCH_TOOL["input_schema"]["required"]
    # filters must be enums so a planner cannot invent a value
    assert "enum" in props["topic"]


def test_tool_topic_enum_matches_the_corpus(index):
    """A topic in the enum that does not exist in the corpus silently
    returns zero results -- the exact bug this lab teaches about."""
    corpus_topics = {d["topic"] for d in index.docs}
    for t in SEARCH_TOOL["input_schema"]["properties"]["topic"]["enum"]:
        assert t in corpus_topics, f"enum topic {t!r} absent from corpus"


def test_planner_splits_compound_questions():
    plan = RuleBasedPlanner(budget=16).plan("mở rộng theo lưu lượng và cân bằng tải giữa region")
    assert len(plan) == 2


def test_planner_keeps_simple_questions_whole():
    assert len(RuleBasedPlanner(budget=16).plan("cân bằng tải giữa nhiều region")) == 1


def test_budget_is_split_not_multiplied():
    """If this breaks, the NB6 comparison silently becomes unfair."""
    budget = 16
    plan = RuleBasedPlanner(budget=budget).plan("mở rộng theo lưu lượng và cân bằng tải")
    assert sum(a.top_k for a in plan) <= budget
    assert sum(a.top_k for a in SingleShotPlanner(budget=budget).plan("x")) <= budget


def test_use_filters_false_emits_no_filters():
    plan = RuleBasedPlanner(budget=8, use_filters=False).plan("cân bằng tải giữa nhiều region")
    assert all(a.topic is None and a.since_year is None for a in plan)


def test_three_intents_receive_the_entire_budget():
    plan = RuleBasedPlanner(budget=16).plan("cân bằng tải; bảo mật dữ liệu; tối ưu truy vấn")
    assert len(plan) == 3
    assert sum(args.top_k for args in plan) == 16


def test_more_intents_than_slots_preserves_question_without_overspending():
    plan = RuleBasedPlanner(budget=2).plan("cân bằng tải; bảo mật dữ liệu; tối ưu truy vấn")
    assert sum(args.top_k for args in plan) == 2
    assert "tối ưu truy vấn" in " ".join(args.query for args in plan)


def test_agent_relaxes_a_starving_filter(index):
    tool = RetrievalTool(index)

    class Starving:
        def plan(self, q):
            return [ToolArgs(query=q, topic="networking", since_year=2027, top_k=5)]

    res = Agent(tool, Starving(), min_evidence=3).answer("cân bằng tải giữa nhiều region")
    assert res.n_calls == 2          # first starved, retried relaxed
    assert len(res.doc_ids) > 0


def test_build_context_runs_without_feast(index):
    from app.agent import build_context
    ctx = build_context("u_001", "tối ưu chi phí", RetrievalTool(index), feature_store=None)
    assert ctx["doc_ids"] and ctx["features"] == {}
