"""Compare a check run with the baseline runs: the knowledge-boundary test (B2) and the fingerprint summary (F1)."""
import statistics

from .questions import norm
from .stats import budget_spent, check_alpha, clopper_pearson, jsd, sign_test_greater


def changed(run_a, run_b, ids):
    return [i for i in ids if norm(run_a["answers"][i]) != norm(run_b["answers"][i])]


def evaluate(baselines, check, number, alpha=0.05, tested_before=0):
    """number names the check; tested_before counts earlier checks that ran a test, which sets the budget
    share. A check without a verdict tests nothing, so it spends no budget."""
    ids = sorted(set.intersection(*(set(r["answers"]) for r in baselines + [check])))
    n = len(ids)
    moved = changed(baselines[0], check, ids)
    x = len(moved)
    result = {
        "check_number": number,
        "items": n,
        "changed": x,
        "changed_rate": x / n,
        "changed_ci": clopper_pearson(x, n),
        "alpha_total": alpha,
        "test_number": tested_before + 1,
        "alpha_this_check": check_alpha(tested_before + 1, alpha),
        "alpha_spent": budget_spent(tested_before + 1, alpha),
        "baseline_runs": len(baselines),
        "examples": [{"id": i, "prompt": baselines[0]["prompts"][i], "baseline": baselines[0]["answers"][i],
                      "now": check["answers"][i]} for i in moved[:8]],
    }
    if len(baselines) < 2:
        result.update(alpha_this_check=0.0, alpha_spent=budget_spent(tested_before, alpha), test_number=None,
                      verdict="no verdict", reason="one baseline run cannot measure the provider's own noise; "
                                                   "run `tracta baseline` again, ideally on another day")
        return result
    # Paired exact test. Anchor on baseline 1; compare the check run and the latest other baseline run
    # with it, question by question. If the provider has not changed, those two runs are exchangeable,
    # so on each question where exactly one of them disagrees with the anchor, either is equally likely
    # to be the one. The count of "check disagrees, baseline agrees" is then Binomial(discordant, 1/2),
    # whatever each question's own noise rate is.
    other = baselines[-1]
    check_moved = set(moved)
    noise_moved = set(changed(baselines[0], other, ids))
    only_check, only_noise = len(check_moved - noise_moved), len(noise_moved - check_moved)
    p = sign_test_greater(only_check, only_check + only_noise)
    result.update(null_changed=len(noise_moved), null_items=n, null_rate=len(noise_moved) / n,
                  null_ci=clopper_pearson(len(noise_moved), n), null_run=other.get("number", len(baselines)),
                  only_check=only_check, only_noise=only_noise, p_value=p,
                  verdict="changed" if p < result["alpha_this_check"] else "no change detected")
    result["fingerprint"] = fingerprint(baselines, check)
    return result


def fingerprint(baselines, check):
    """Median Jensen-Shannon divergence of first-token distributions: check against baseline 1, and
    baseline 2 against baseline 1 (the noise). Supporting evidence only; it does not decide."""
    b1, b2 = baselines[0].get("tokens"), baselines[1].get("tokens")
    now = check.get("tokens")
    if not (b1 and b2 and now):
        return None
    probes = sorted(set(b1) & set(b2) & set(now))
    if not probes:
        return None
    return {"probes": len(probes),
            "median_jsd_check": statistics.median(jsd(now[p], b1[p]) for p in probes),
            "median_jsd_noise": statistics.median(jsd(b2[p], b1[p]) for p in probes)}
