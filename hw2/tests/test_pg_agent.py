#!/usr/bin/env python

import pytest
import cs285.agents.pg_agent as pg_agent


@pytest.mark.parametrize(
    "rewards, discount_factor, correct_answer",
    [
        ([1, 2, 4], 0.5, [3, 3, 3]),
    ],
)
def test_discounted_return(rewards, discount_factor, correct_answer):
    ret = pg_agent.discounted_return(rewards, discount_factor)
    assert ret == correct_answer


@pytest.mark.parametrize(
    "rewards, discount_factor, correct_answer",
    [
        ([1, 2, 4], 0.5, [3, 4, 4]),
        ([1, 2, 3], 1, [6, 5, 3]),
    ],
)
def test_discounted_cumsum(rewards, discount_factor, correct_answer):
    ret = pg_agent.discounted_cumsum(rewards, discount_factor)
    assert ret == correct_answer
