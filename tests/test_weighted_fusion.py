from hybrid_retrieval.search import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_can_bias_toward_lexical_rankings():
    fused = reciprocal_rank_fusion(
        [["lexical_top", "semantic_top"], ["semantic_top", "lexical_top"]],
        weights=[1.5, 1.0],
        k=60,
    )

    assert fused[:2] == ["lexical_top", "semantic_top"]


def test_reciprocal_rank_fusion_keeps_equal_weight_behavior_by_default():
    fused = reciprocal_rank_fusion(
        [["shared", "lexical_only"], ["shared", "semantic_only"]],
        k=60,
    )

    assert fused[0] == "shared"
