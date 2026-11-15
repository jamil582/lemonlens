"""Synthetic complaints for offline demo and tests.

Makes, models and narratives here are invented. Nothing in this file is real
NHTSA data or describes a real vehicle.
"""
from __future__ import annotations

import random

import pandas as pd

from .nhtsa import frames

MAKE = "DEMO MOTORS"
MODELS = {"ARROW": (2019, 2022), "BISON": (2020, 2023)}

PATTERNS = [
    ("SERVICE BRAKES", 0.07, 0.0, [
        "The brake pedal went soft and sank to the floor while slowing for a red light. Stopping distance was much longer than normal.",
        "Automatic emergency braking activated with nothing in front of the vehicle on the highway. The car behind nearly hit me.",
        "Grinding noise from the front brakes at {miles} miles. The dealer said the rotors were warped and it is not covered.",
    ]),
    ("ENGINE", 0.03, 0.02, [
        "Strong smell of fuel in the oil and the oil level keeps rising. The dealer says the gasoline mixing with oil is normal.",
        "Engine stalled without warning while merging onto the freeway at {speed} mph. All dash lights came on and power steering was lost.",
        "Check engine light came on and the engine went into limp mode. It hesitates and loses power going uphill.",
    ]),
    ("ELECTRICAL SYSTEM", 0.02, 0.05, [
        "The battery drains overnight and the vehicle will not start. It has been jump started four times and the battery replaced twice.",
        "The infotainment screen goes black while driving and the backup camera does not display when reversing.",
        "Burning smell from under the dashboard and smoke from the fuse box area after parking in the garage.",
    ]),
    ("POWER TRAIN", 0.05, 0.0, [
        "Transmission jerks and shudders between second and third gear. It hesitates when accelerating from a stop.",
        "The vehicle lurched forward while in park with the engine running. The transmission was later replaced at {miles} miles.",
    ]),
    ("STEERING", 0.09, 0.0, [
        "Steering wheel locked up momentarily while driving at {speed} mph and the vehicle drifted toward the next lane.",
        "Steering feels sticky at highway speed and needs constant correction to stay in the lane.",
    ]),
    ("AIR BAGS", 0.25, 0.0, [
        "The air bag warning light stays on. The dealer says the passenger seat sensor has failed and the part is backordered.",
        "In a frontal collision at about {speed} mph the air bags did not deploy. The driver had chest injuries.",
    ]),
    ("FUEL/PROPULSION SYSTEM", 0.02, 0.04, [
        "The fuel pump failed and the vehicle shut off in traffic. It was towed and the pump was replaced.",
        "Strong gasoline smell inside the cabin after refuelling. A leak was found near the fuel tank.",
    ]),
    ("VISIBILITY/WIPER", 0.01, 0.0, [
        "The windshield cracked on its own overnight with no impact. This is the second windshield.",
        "Wipers stopped working during heavy rain and the driver could not see the road.",
    ]),
]
WEIGHTS = {"ARROW": [10, 22, 12, 9, 4, 5, 4, 3], "BISON": [18, 6, 9, 14, 8, 4, 6, 3]}


def complaints(n: int = 600, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        model = rng.choice(list(MODELS))
        lo, hi = MODELS[model]
        year = rng.randint(lo, hi)
        component, crash_p, fire_p, texts = rng.choices(PATTERNS, weights=WEIGHTS[model])[0]
        filed = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=rng.randint(120, 1500))
        summary = rng.choice(texts).format(miles=rng.randrange(8, 70) * 1000, speed=rng.randrange(25, 75, 5))
        extra = rng.choice([
            "", " The manufacturer was contacted and offered no assistance.",
            " The failure has happened more than once.", " The vehicle was taken to the dealer who could not reproduce the problem.",
        ])
        crash = rng.random() < crash_p
        comps = [component] + (["UNKNOWN OR OTHER"] if rng.random() < 0.08 else [])
        rows.append(
            {
                "odi_number": str(90000000 + i),
                "make": MAKE,
                "model": model,
                "model_year": year,
                "vehicle": f"{MAKE} {model}",
                "components": comps,
                "component": component,
                "date_filed": filed,
                "date_incident": filed - pd.Timedelta(days=rng.randint(1, 60)),
                "crash": crash,
                "fire": rng.random() < fire_p,
                "injuries": rng.randint(1, 2) if crash and rng.random() < 0.4 else 0,
                "deaths": 0,
                "summary": summary + extra,
            }
        )
    recalls = [
        {
            "campaign": "DEMO-001", "make": MAKE, "model": "ARROW", "model_year": y, "component": "ENGINE",
            "date_received": pd.Timestamp("2022-03-14"),
            "summary": "Fuel may dilute the engine oil during short trips in cold weather.",
            "consequence": "Oil dilution can cause engine stalling, increasing the risk of a crash.",
            "remedy": "Dealers will update the engine control software, free of charge.",
        }
        for y in (2019, 2020)
    ] + [
        {
            "campaign": "DEMO-002", "make": MAKE, "model": "BISON", "model_year": 2021, "component": "SERVICE BRAKES",
            "date_received": pd.Timestamp("2023-06-02"),
            "summary": "The automatic emergency braking system may activate unexpectedly.",
            "consequence": "Unexpected braking increases the risk of a rear-end collision.",
            "remedy": "Dealers will recalibrate the forward radar sensor, free of charge.",
        }
    ]
    return frames(rows, recalls)
