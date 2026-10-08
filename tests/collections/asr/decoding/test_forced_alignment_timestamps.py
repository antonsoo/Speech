# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import get_forced_aligned_timestamps_with_external_model
from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer


@pytest.mark.unit
@pytest.mark.parametrize("punctuation", [".", ","])
@pytest.mark.parametrize("trailing_punctuation", ["", "."])
@pytest.mark.parametrize("timestamp_type", ["all", "segment"])
def test_segment_end_with_internal_punctuation(tmp_path, punctuation, trailing_punctuation, timestamp_type):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("'a'\n'b'\n'.'\n','\n", encoding="utf-8")
    tokenizer = CharTokenizer(str(vocab_path), unk_token="?", special_tokens_to_remove_while_decoding=[])
    model = SimpleNamespace(
        tokenizer=tokenizer,
        cfg={"preprocessor": {"window_stride": 0.01}},
        encoder=SimpleNamespace(subsampling_factor=8),
        device=torch.device("cpu"),
    )
    text = "a" + punctuation + "b" + trailing_punctuation
    blank_id = tokenizer.vocab_size
    frames = torch.full((40,), blank_id)
    for token, start in [("a", 10), (punctuation, 20), ("b", 30)]:
        frames[start : start + 2] = tokenizer.token_to_id(token)
    if trailing_punctuation:
        frames[34:36] = tokenizer.token_to_id(trailing_punctuation)
    log_probs = torch.full((40, blank_id + 1), -100.0)
    log_probs.scatter_(1, frames[:, None], 0.0)

    result = get_forced_aligned_timestamps_with_external_model(
        audio=[Hypothesis(score=0.0, y_sequence=log_probs, text=text)],
        external_ctc_model=model,
        main_model_predictions=[Hypothesis(score=0.0, y_sequence=[], text=text)],
        has_hypotheses=True,
        timestamp_type=timestamp_type,
        verbose=False,
    )[0]

    # Internal punctuation does not end the segment. A trailing period still
    # ends at the preceding speech token, which occupies frames 30 and 31.
    segment = result.timestamp["segment"][0]
    assert segment["segment"] == text
    assert segment["start_offset"] == 10
    assert segment["end_offset"] == 32
    assert segment["start"] == pytest.approx(0.8)
    assert segment["end"] == pytest.approx(2.56)
    if timestamp_type == "all":
        assert segment["end_offset"] == result.timestamp["word"][-1]["end_offset"]
        assert segment["end"] == result.timestamp["word"][-1]["end"]
