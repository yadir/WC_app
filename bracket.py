"""
bracket.py — the REAL 2026 knockout structure, transcribed from the official
FIFA / Fox Sports schedule (104 matches).

Two parts:
  1. R32_SLOTS: the 16 Round-of-32 matches (M73-M88), each as a pair of "slots".
     A slot is one of:
        ("WIN", "C")        -> winner of group C
        ("RUN", "A")        -> runner-up of group A
        ("THIRD", ["C","E","F","H","I"])  -> a 3rd-place team from one of these
  2. TREE: which match feeds which, M89 (R16) up to M104 (Final).

The third-place slots list their *candidate* groups exactly as the schedule
shows them. We assign the 8 qualifying third-place teams to these slots with a
constraint solver (see simulate.py) so each 3rd team lands in a slot whose
candidate list includes its group — which is what FIFA's Annex C table
guarantees.
"""

# Each entry: match_id -> (slotA, slotB)
R32_SLOTS = {
    "M73": (("RUN", "A"), ("RUN", "B")),
    "M74": (("WIN", "E"), ("THIRD", ["A", "B", "C", "D", "F"])),
    "M75": (("WIN", "F"), ("RUN", "C")),
    "M76": (("WIN", "C"), ("RUN", "F")),
    "M77": (("WIN", "I"), ("THIRD", ["C", "D", "F", "G", "H"])),
    "M78": (("RUN", "E"), ("RUN", "I")),
    "M79": (("WIN", "A"), ("THIRD", ["C", "E", "F", "H", "I"])),
    "M80": (("WIN", "L"), ("THIRD", ["E", "H", "I", "J", "K"])),
    "M81": (("WIN", "D"), ("THIRD", ["B", "E", "F", "I", "J"])),
    "M82": (("WIN", "G"), ("THIRD", ["A", "E", "H", "I", "J"])),
    "M83": (("RUN", "K"), ("RUN", "L")),
    "M84": (("WIN", "H"), ("RUN", "J")),
    "M85": (("WIN", "B"), ("THIRD", ["E", "F", "G", "I", "J"])),
    "M86": (("WIN", "J"), ("RUN", "H")),
    "M87": (("WIN", "K"), ("THIRD", ["D", "E", "I", "J", "L"])),
    "M88": (("RUN", "D"), ("RUN", "G")),
}

# Knockout tree: match_id -> (feeder_match_A, feeder_match_B)
# Winners of the two feeders meet in this match.
TREE = {
    # Round of 16
    "M89": ("M74", "M77"),
    "M90": ("M73", "M75"),
    "M91": ("M76", "M78"),
    "M92": ("M79", "M80"),
    "M93": ("M83", "M84"),
    "M94": ("M81", "M82"),
    "M95": ("M86", "M88"),
    "M96": ("M85", "M87"),
    # Quarter-finals
    "M97": ("M89", "M90"),
    "M98": ("M93", "M94"),
    "M99": ("M91", "M92"),
    "M100": ("M95", "M96"),
    # Semi-finals
    "M101": ("M97", "M98"),
    "M102": ("M99", "M100"),
    # Final (3rd-place match M103 is omitted from sim; not needed for our odds)
    "M104": ("M101", "M102"),
}

FINAL_MATCH = "M104"

# Which round each match belongs to (for tagging how far a team got)
ROUND_OF = {}
for m in R32_SLOTS: ROUND_OF[m] = "R32"
for m in ["M89","M90","M91","M92","M93","M94","M95","M96"]: ROUND_OF[m] = "R16"
for m in ["M97","M98","M99","M100"]: ROUND_OF[m] = "QF"
for m in ["M101","M102"]: ROUND_OF[m] = "SF"
ROUND_OF["M104"] = "F"
