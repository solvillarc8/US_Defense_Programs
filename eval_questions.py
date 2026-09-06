"""
eval_questions.py — the evaluation harness's test set.

Every fact below was pulled from REAL tool output generated while building
this project tonight — not guessed — so every case is genuinely answerable
from the ingested corpora.

Case types (evaluate.py checks in this priority order):
  expect_refusal      -> answer should contain the refusal phrase
  expect_both_sources -> citations should include BOTH GAO and Weapons Book
  expect_keywords     -> every keyword must appear in the answer (case-insensitive)
    + optional expect_page -> at least one citation must cover this page
"""

EVAL_QUESTIONS = [
    # --- GAO: CVN 78 (heavily verified tonight) ---------------------------
    {"id": "gao-cvn78-cvn79-cost", "question": "How much have CVN 79 costs increased since 2021, according to GAO?",
     "expect_keywords": ["1.5 billion"], "expect_page": 145},
    {"id": "gao-cvn78-cvn80-cost", "question": "According to GAO, how much did CVN 80 costs increase?",
     "expect_keywords": ["500 million"], "expect_page": 145},
    {"id": "gao-cvn78-delay", "question": "Why was CVN 79's delivery delayed, according to GAO?",
     "expect_keywords": ["elevator"], "expect_page": 145},
    {"id": "gao-cvn78-testing", "question": "What is the new completion date for CVN 78's initial operational testing?",
     "expect_keywords": ["2028"], "expect_page": 145},
    {"id": "gao-cvn78-contractor", "question": "Who is the prime contractor for the CVN 78 program?",
     "expect_keywords": ["Huntington Ingalls"], "expect_page": 145},
    {"id": "gao-cvn78-class", "question": "What class of aircraft carrier is CVN 78?",
     "expect_keywords": ["Ford"], "expect_page": 145},

    # --- GAO: sensor programs (verified) ------------------------------------
    {"id": "gao-hades", "question": "What does the HADES program integrate, according to GAO?",
     "expect_keywords": ["multi-intelligence"], "expect_page": 119},
    {"id": "gao-f22see", "question": "What does the F-22 SeE program add to the F-22 aircraft?",
     "expect_keywords": ["sensor"], "expect_page": 97},
    {"id": "gao-sdb2", "question": "What types of sensors does the SDB II weapon use, according to GAO?",
     "expect_keywords": ["laser"], "expect_page": 108},

    # --- GAO: hypersonics (verified) ----------------------------------------
    {"id": "gao-cps-page", "question": "What page does GAO's CPS hypersonic weapon profile appear on?",
     "expect_keywords": ["CPS"], "expect_page": 143},
    {"id": "gao-hacm-page", "question": "What page does GAO's HACM hypersonic weapon profile appear on?",
     "expect_keywords": ["HACM"], "expect_page": 99},

    # --- GAO: presence-only checks (safe, no page asserted) ----------------
    {"id": "gao-mq25", "question": "Does GAO's report include a profile for the MQ-25 Stingray program?",
     "expect_keywords": ["MQ-25"]},
    {"id": "gao-t7a", "question": "What GAO program is nicknamed 'Red Hawk'?",
     "expect_keywords": ["T-7A"]},
    {"id": "gao-f22-full", "question": "What is the full name of the F-22 aircraft program GAO assesses?",
     "expect_keywords": ["Raptor"]},

    # --- Weapons Book (verified) ---------------------------------------------
    {"id": "wb-patriot-radar", "question": "What radar does the PATRIOT system use, per DoD's Weapons Book?",
     "expect_keywords": ["AN/MPQ-65"], "expect_page": 62},
    {"id": "wb-ltamds", "question": "What does LTAMDS replace, according to the Weapons Book?",
     "expect_keywords": ["PATRIOT"], "expect_page": 62},
    {"id": "wb-cvn-rcoh", "question": "Which carrier's refueling overhaul does the Weapons Book fully fund in FY2026?",
     "expect_keywords": ["Truman"], "expect_page": 98},
    {"id": "wb-sentinel", "question": "What was the Sentinel missile program formerly called, per the Weapons Book?",
     "expect_keywords": ["Ground Based Strategic Deterrent"], "expect_page": 89},
    {"id": "wb-lrhw", "question": "What is the nickname for the Army's Long Range Hypersonic Weapon, per the Weapons Book?",
     "expect_keywords": ["Dark Eagle"], "expect_page": 114},
    {"id": "wb-hacm", "question": "What service branch's Hypersonic Attack Cruise Missile does the Weapons Book describe on its own dedicated page?",
     "expect_keywords": ["Air Force"], "expect_page": 116},
    {"id": "wb-b21", "question": "What was the B-21 Raider previously referred to as, per the Weapons Book?",
     "expect_keywords": ["Long Range"], "expect_page": 27},
    {"id": "wb-xm30", "question": "What was the Army's XM30 Combat Vehicle formerly called, per the Weapons Book?",
     "expect_keywords": ["Optionally Manned"], "expect_page": 54},
    {"id": "wb-f47", "question": "Does the Weapons Book describe an F-47 acquisition program?",
     "expect_keywords": ["F-47"], "expect_page": 31},

    # --- Cross-source structural checks --------------------------------------
    {"id": "cross-cvn78-compare", "question": "Compare how GAO and DoD's Weapons Book each describe the CVN 78 program.",
     "expect_both_sources": True, "expect_tool": "check_source_divergence"},
    {"id": "cross-sensors", "question": "What weapon programs involve sensors, according to both GAO and DoD's Weapons Book?",
     "expect_both_sources": True, "expect_tool": "check_source_divergence"},

    # --- Multimodal: numbers that ONLY exist on the chart, not clean prose --
    {"id": "chart-cvn78-cost", "question": "What is the Program Cost breakdown (Procurement vs Development) on CVN 78's chart?",
     "expect_keywords": ["86,768", "8,749"], "expect_page": 145},
    {"id": "chart-cvn78-quantities", "question": "According to CVN 78's chart, how many units are in Procurement?",
     "expect_keywords": ["6"], "expect_page": 145},

    # --- Regression test: the "sensors" broad-query bug we fixed tonight ----
    {"id": "regression-broad-sensors", "question": "What sensors will be used in the next programs?",
     "expect_keywords": ["sensor"], "expect_tool": "search_gao_reports"},

    # Exact shape of the bibliography/coverage hallucination found live.
    {"id": "regression-f35-bibliography", "question": "What does GAO say about the F-35?",
     "expect_refusal": True, "expect_tool": "search_gao_reports"},

    # --- CRS: congressional funding and enacted-law coverage -----------------
    {"id": "crs-aircraft-request", "question": "How much did DOD request for aircraft and related systems in FY2026, according to CRS?",
     "expect_keywords": ["68.3 billion"], "expect_page": 10, "expect_tool": "search_crs_reports"},
    {"id": "crs-aircraft-ndaa", "question": "How much discretionary funding did the enacted FY2026 NDAA authorize for selected aircraft, according to CRS?",
     "expect_keywords": ["45.0 billion"], "expect_page": 10, "expect_tool": "search_crs_reports"},
    {"id": "crs-aircraft-appropriations", "question": "How much did the FY2026 appropriations act provide for selected aircraft systems, according to CRS?",
     "expect_keywords": ["49.1 billion"], "expect_page": 10, "expect_tool": "search_crs_reports"},

    # --- Refusals: genuinely out-of-corpus ------------------------------------
    {"id": "refusal-worldcup", "question": "Who won the 2022 FIFA World Cup?", "expect_refusal": True},
    {"id": "refusal-weather", "question": "What's the weather like in Washington DC today?", "expect_refusal": True},
    {"id": "refusal-f35", "question": "What company makes the engine for the F-35?", "expect_refusal": True},
    # ^ F-35 is deliberately tricky: it's a famous, real program, but this
    # specific GAO report doesn't profile it (it has its own separate GAO
    # report) — this tests that the model doesn't fill the gap from general
    # training knowledge just because the program sounds familiar.
    {"id": "refusal-recipe", "question": "Give me a recipe for chocolate chip cookies.", "expect_refusal": True},
]
