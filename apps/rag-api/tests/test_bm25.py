from __future__ import annotations

import pytest

from app.retrieval.bm25 import (
    BM25Index,
    FusedHit,
    fuse_routes,
    fuse_with_quota,
    tokenize,
)


def _index() -> BM25Index:
    return BM25Index.from_records(
        [
            ("a", "M8P515", "IOB5 配置为输出之后仍拉不出有力的高电平，要设置 PUB[5]=1。"),
            ("b", "M8P515", "PWM1 是 8+4 位的脉宽调制模块，PWM1DH 是数据高位。"),
            ("c", "M8P516", "IOB5 的上拉能力由 PUB5 位控制，输出高电平较弱。"),
            ("d", "M8P516", "ADC 转换完成后 ADCON0 的 GO 位自动清零。"),
            ("e", "M8P516", "PWM 波形图的两路高电平宽度不同。"),
        ]
    )


# ---------------------------------------------------------------- tokenization


def test_tokenize_nfkc_and_lowercase() -> None:
    assert tokenize("ＡＤＣＯＮ０") == ["adcon0"]
    assert tokenize("PWM1DH") == ["pwm1dh"]
    assert tokenize("M8P515") == ["m8p515"]


def test_tokenize_cjk_bigrams_in_order() -> None:
    assert tokenize("脉宽调制") == ["脉宽", "宽调", "调制"]


def test_tokenize_single_cjk_character_is_a_unigram() -> None:
    assert tokenize("图") == ["图"]


def test_tokenize_mixed_ascii_and_cjk_keeps_identifiers_whole() -> None:
    assert tokenize("PWM1DH 是数据高位") == ["pwm1dh", "是数", "数据", "据高", "高位"]


def test_tokenize_keeps_values_and_register_names() -> None:
    tokens = tokenize("8+4 位, 0.000V, PUB[5]=1")
    assert "8+4" in tokens
    assert "0.000v" in tokens
    assert "pub" in tokens
    assert "5" in tokens


def test_tokenize_empty_and_whitespace_only() -> None:
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []


# ----------------------------------------------------------------- BM25 search


def test_search_ranks_the_lexically_matching_chunk_first() -> None:
    hits = _index().search("PUB5 位设置之后输出高电平", "M8P515")
    assert hits
    assert hits[0].chunk_id == "a"
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_search_filters_by_product_before_scoring() -> None:
    index = _index()
    assert [hit.chunk_id for hit in index.search("ADCON0", "M8P516")] == ["d"]
    assert index.search("ADCON0", "M8P515") == []
    # A token shared by two products only returns chunks inside the scope.
    assert {hit.chunk_id for hit in index.search("高电平", "M8P516")} == {"c", "e"}
    assert {hit.chunk_id for hit in index.search("高电平", "M8P515")} == {"a"}


def test_search_accepts_a_product_sequence_for_multi_source_questions() -> None:
    hits = _index().search("IOB5", ("M8P515", "M8P516"))
    assert {hit.chunk_id for hit in hits} == {"a", "c"}


def test_search_unknown_product_returns_empty() -> None:
    assert _index().search("PWM", "M9F999") == []


def test_search_without_a_matching_token_returns_empty() -> None:
    assert _index().search("完全不相关的查询词", "M8P515") == []


def test_search_honours_top_k_and_ties_break_on_chunk_id() -> None:
    index = BM25Index.from_records(
        [("z", "P", "相同文本"), ("a", "P", "相同文本"), ("m", "P", "相同文本")]
    )
    hits = index.search("相同文本", "P", top_k=2)
    assert [hit.chunk_id for hit in hits] == ["a", "m"]


def test_search_empty_inputs_return_empty() -> None:
    index = _index()
    assert index.search("", "M8P515") == []
    assert index.search("PWM", "M8P515", top_k=0) == []
    assert BM25Index.from_records([]).search("PWM", "M8P515") == []


def test_duplicate_chunk_ids_are_indexed_once() -> None:
    index = BM25Index.from_records([("a", "P", "PWM"), ("a", "P", "PWM")])
    assert len(index) == 1
    assert index.search("PWM", "P")[0].chunk_id == "a"


def test_index_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        BM25Index.from_records([], k1=-1)
    with pytest.raises(ValueError):
        BM25Index.from_records([], b=1.5)


# -------------------------------------------------------------------- fusion


def test_fuse_routes_orders_a_document_found_by_both_routes_first() -> None:
    dense = ["d1", "d2", "d3"]
    sparse = ["s1", "d2"]
    fused = fuse_routes(dense, sparse)
    assert fused[0].chunk_id == "d2"
    assert fused[0].routes == ("dense", "sparse")
    # d2 gets both ranks; d1 and s1 tie at 1/61 and break on chunk id; d3 is 1/63.
    assert [hit.chunk_id for hit in fused] == ["d2", "d1", "s1", "d3"]
    assert all(isinstance(hit, FusedHit) for hit in fused)


def test_fuse_routes_scores_follow_the_rrf_formula() -> None:
    fused = fuse_routes(["x", "y"], [], k0=60)
    assert fused[0].score == pytest.approx(1.0 / 61)
    assert fused[1].score == pytest.approx(1.0 / 62)


def test_fuse_routes_supports_weights() -> None:
    heavy_sparse = fuse_routes(["d1"], ["s1"], weights=(0.0, 2.0))
    assert heavy_sparse[0].chunk_id == "s1"
    assert heavy_sparse[0].score == pytest.approx(2.0 / 61)


def test_fuse_routes_dedupes_within_a_route_and_honours_limit() -> None:
    fused = fuse_routes(["a", "a", "b"], ["b"])
    assert [hit.chunk_id for hit in fused] == ["b", "a"]
    assert len(fuse_routes(["a"], ["b"], limit=1)) == 1
    assert fuse_routes(["a"], ["b"], limit=0) == []


def test_fuse_routes_empty_inputs() -> None:
    assert fuse_routes([], []) == []
    assert [hit.chunk_id for hit in fuse_routes([], ["s1"])] == ["s1"]
    assert [hit.chunk_id for hit in fuse_routes(["d1"], [])] == ["d1"]


def test_fuse_routes_accepts_objects_with_chunk_id() -> None:
    class Hit:
        def __init__(self, chunk_id: str) -> None:
            self.chunk_id = chunk_id

    fused = fuse_routes([Hit("a"), Hit("b")], [Hit("b")])
    assert [hit.chunk_id for hit in fused] == ["b", "a"]


def test_fuse_routes_rejects_bad_options() -> None:
    with pytest.raises(ValueError):
        fuse_routes([], [], k0=-1)
    with pytest.raises(ValueError):
        fuse_routes([], [], weights=(1.0,))
    with pytest.raises(ValueError):
        fuse_routes([], [], weights=(1.0, float("nan")))


# --------------------------------------------------------------- route quota


def test_quota_reserves_bm25_top2_and_dense_top3_before_rrf_fill() -> None:
    dense = ["d1", "d2", "d3", "d4", "d5", "d6"]
    sparse = ["s1", "s2"]
    fused = fuse_with_quota(dense, sparse, dense_slots=3, sparse_slots=2, limit=5)
    # Reserved head: d1/s1 (1/61), d2/s2 (1/62), d3 (1/63); ties break on chunk_id.
    assert [hit.chunk_id for hit in fused] == ["d1", "s1", "d2", "s2", "d3"]


def test_quota_keeps_a_low_weight_route_in_the_head() -> None:
    dense = ["d1", "d2", "d3", "d4", "d5"]
    sparse = ["s1", "s2"]
    weights = (1.0, 0.25)
    without_quota = fuse_routes(dense, sparse, weights=weights)
    assert {hit.chunk_id for hit in without_quota[:5]} == {"d1", "d2", "d3", "d4", "d5"}
    with_quota = fuse_with_quota(dense, sparse, dense_slots=3, sparse_slots=2, weights=weights)
    assert [hit.chunk_id for hit in with_quota[:5]] == ["d1", "d2", "d3", "s1", "s2"]


def test_quota_fills_the_rest_in_rrf_order() -> None:
    dense = ["d1", "d2", "d3", "d4", "d5"]
    sparse = ["s1", "s2", "d4"]
    fused = fuse_with_quota(dense, sparse, dense_slots=2, sparse_slots=1, limit=6)
    # Reserved: d1, s1, d2 (scores 1/61, 1/61, 1/62; d1 < s1 on the tie).
    # Then plain-RRF order: d4 (dense#4 + sparse#3), s2 (1/62), d3 (1/63).
    assert [hit.chunk_id for hit in fused] == ["d1", "s1", "d2", "d4", "s2", "d3"]


def test_quota_dedupes_route_overlap_and_keeps_head_size() -> None:
    fused = fuse_with_quota(["a", "b", "c"], ["c", "a"])
    ids = [hit.chunk_id for hit in fused]
    assert len(ids) == len(set(ids))
    # Reserved: dense a,b + sparse c,a -> all three; RRF favours a then c then b.
    assert ids == ["a", "c", "b"]


def test_quota_zero_slots_equals_plain_rrf() -> None:
    dense = ["d1", "d2"]
    sparse = ["s1"]
    assert [
        hit.chunk_id for hit in fuse_with_quota(dense, sparse, dense_slots=0, sparse_slots=0)
    ] == [hit.chunk_id for hit in fuse_routes(dense, sparse)]


def test_quota_empty_inputs_and_limits() -> None:
    assert fuse_with_quota([], []) == []
    assert [hit.chunk_id for hit in fuse_with_quota([], ["s1"], limit=1)] == ["s1"]
    assert fuse_with_quota(["a"], ["b"], limit=0) == []
    with pytest.raises(ValueError):
        fuse_with_quota([], [], dense_slots=-1)
