"""Drive synthetic listeners at the API so the bandit has something to learn from.

The experiments dashboard is only interesting once arms have separated, and
that needs real outcomes. Without this, anyone running the project locally sees
four arms sitting at their Beta(1,1) prior and has no way to tell whether the
bandit works.

Each simulated listener has a hidden preference — a true reward rate per arm —
that the bandit is not told. A correct implementation should concentrate
traffic on the best arm without ever being handed the answer, which is exactly
what this script checks.

    python -m scripts.simulate_traffic --sessions 400

Writes real rows to `interaction_events`. Point it at a throwaway database if
you care about the event log.
"""
import argparse
import json
import logging
import random
import time
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("simulate")

# Ground truth the bandit never sees. `discovery` is genuinely the best arm
# here; if the posteriors don't find it, something is wrong.
TRUE_REWARD_RATE = {
    "control": 0.20,
    "descriptor_heavy": 0.25,
    "diverse": 0.30,
    "discovery": 0.45,
}


def _get(base: str, path: str, ip: str):
    req = urllib.request.Request(base + path, headers={"X-Forwarded-For": ip})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _post(base: str, path: str, payload: dict, ip: str) -> None:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Forwarded-For": ip},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=30).close()


# Reads are budgeted at 120/min per IP. The simulator paces itself under that
# rather than rotating addresses to slip past it: a load generator that evades
# the service's own protections measures a system nobody actually runs.
READ_INTERVAL_SEC = 60.0 / 110


def simulate(base: str, sessions: int, per_session: int, seed: int) -> None:
    rng = random.Random(seed)

    # A distinct forwarded address per run keeps the simulator inside its own
    # rate-limit bucket instead of starving real traffic.
    ip = f"10.200.{rng.randint(1, 254)}.{rng.randint(1, 254)}"

    catalog = _get(base, "/tracks?limit=100", ip)
    if not catalog:
        raise SystemExit("catalog is empty — ingest tracks first")
    seed_ids = [t["id"] for t in catalog]

    served: dict[str, int] = {}

    for i in range(sessions):
        time.sleep(READ_INTERVAL_SEC)
        track_id = rng.choice(seed_ids)
        try:
            body = _get(base, f"/recommend/track/{track_id}?top_k=5&session_id=sim-{i}", ip)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Shouldn't happen with pacing, but back off rather than die.
                logger.warning("rate limited at session %d, backing off", i)
                time.sleep(30)
                continue
            raise

        variant = body["variant"]
        served[variant] = served.get(variant, 0) + 1

        for position, rec in enumerate(body["results"][:per_session]):
            rewarded = rng.random() < TRUE_REWARD_RATE.get(variant, 0.25)
            _post(
                base,
                "/events",
                {
                    "session_id": f"sim-{i}",
                    "variant": variant,
                    "track_id": rec["track"]["id"],
                    "seed_track_id": track_id,
                    "action": "complete" if rewarded else "skip",
                    "position": position,
                    # Below the skip threshold, so a non-reward actually counts
                    # against the arm rather than being treated as a late skip.
                    "dwell_ms": 180_000 if rewarded else 2_500,
                },
                ip,
            )

        if (i + 1) % 50 == 0:
            logger.info("%d/%d sessions", i + 1, sessions)

    report = _get(base, "/experiments/report", ip)

    print()
    print("traffic served per arm:", dict(sorted(served.items(), key=lambda kv: -kv[1])))
    print()
    print(f"{'arm':<20}{'true rate':>11}{'estimated':>11}{'P(best)':>10}{'served':>9}")
    print("-" * 61)
    for arm in report["arms"]:
        true = TRUE_REWARD_RATE.get(arm["variant"])
        print(
            f"{arm['variant']:<20}"
            f"{(f'{true:.2f}' if true else '—'):>11}"
            f"{arm['posterior_mean']:>11.3f}"
            f"{(arm['probability_best'] or 0):>10.3f}"
            f"{served.get(arm['variant'], 0):>9}"
        )
    print()
    best = max(TRUE_REWARD_RATE, key=lambda k: TRUE_REWARD_RATE[k])
    leader = max(report["arms"], key=lambda a: a["probability_best"] or 0)
    verdict = "found it" if leader["variant"] == best else "MISSED"
    print(f"best arm is '{best}'; bandit favours '{leader['variant']}' — {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate listener traffic")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--sessions", type=int, default=300)
    parser.add_argument("--per-session", type=int, default=2, help="outcomes logged per session")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    simulate(args.base, args.sessions, args.per_session, args.seed)


if __name__ == "__main__":
    main()
