"""In-process reference strategies (backend: baseline). No API calls, used for tests and as
reference points in analyses.

  follow_g           predicts what the explanation's rule says; without an explanation, the reference label
  true_f             predicts the true rule (oracle)
  copy_reference     always the reference decision
  majority_evidence  the majority label of the evidence; the reference decision when there is none
"""

from __future__ import annotations

from collections import Counter

from nsg_sim.backends.base import Backend, Request, Response, register


@register("baseline")
class Baseline(Backend):
    calls_api = False
    STRATEGIES = ("follow_g", "true_f", "copy_reference", "majority_evidence")

    def __init__(self, sim, run_settings=None):
        super().__init__(sim, run_settings)
        if sim.model not in self.STRATEGIES:
            raise ValueError(f"baseline {sim.model!r}: choose from {self.STRATEGIES}")

    async def complete(self, request: Request) -> Response:
        r = request.record
        reference = r["label_x"]
        if self.sim.model == "follow_g":
            answer = r["explanation_label_cx"] if request.with_explanation else reference
        elif self.sim.model == "true_f":
            answer = r["label_cx"]
        elif self.sim.model == "majority_evidence":
            labels = Counter(e["label"] for e in r.get("evidence") or [])
            answer = labels.most_common(1)[0][0] if labels else reference
        else:
            answer = reference
        return Response(text=f"[ANSWER]\n{answer}")
