# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 Felipe M. Muniz
#
# This file is part of Aletheion Core (Educational Release).
#
# Aletheion Core is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# For commercial licensing inquiries, contact: licensing@aletheiaengine.dev

"""
Aletheion Core - Educational Release

A simplified implementation of the core symbolic coherence monitoring system
described in the paper "Aletheion: A Semi-Symbolic Architecture for Internal
Coherence Monitoring in Neural Language Systems".

This educational release contains:
- Q metric: Epistemic quality measure in [0, 1]
- VARO operator: Variational Anti-Resonance Operator for state updates
- Epistemic gating: Quality threshold enforcement
"""

from .epistemic_gate import check_quality_threshold, epistemic_gate
from .q_metric import cosine_similarity, q_metric
from .varo import normalize_vector, varo_update

__version__ = "1.0.0-edu"
__all__ = [
    "q_metric",
    "cosine_similarity",
    "varo_update",
    "normalize_vector",
    "epistemic_gate",
    "check_quality_threshold",
]
