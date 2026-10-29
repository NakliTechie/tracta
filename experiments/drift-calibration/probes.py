"""Fixed probe sets for the drift and fingerprint calibration.

Three sets, one per hypothesis in plan/research/2026-10-27-detection-hypotheses.md:
  open_prompts()     B1, rank-uniformity drift: short open-ended prompts, sampled answers
  boundary_prompts() B2, knowledge-boundary drift: short-answer facts and sums, greedy answers
  token_probes()     F1, single-token fingerprint: one-token answers, top log-probabilities

Every set is deterministic, so two runs on different days send identical requests.
"""
import random

SUBJECTS = [
    "rivers", "bread", "volcanoes", "libraries", "trains", "honeybees", "winter", "chess",
    "bridges", "coffee", "deserts", "lighthouses", "music", "the moon", "gardens", "clocks",
    "maps", "rain", "markets", "bicycles",
]

TEMPLATES = [
    "Write one sentence about {s}.",
    "Give one surprising fact about {s}.",
    "Describe {s} to a child in one sentence.",
    "Write a short slogan about {s}.",
    "What is one common misconception about {s}? Answer in one sentence.",
    "Write the first line of a poem about {s}.",
    "Name one way {s} changed over the last century, in one sentence.",
    "Give one tip for someone learning about {s}.",
    "Write a one-sentence news headline about {s}.",
    "Explain why people care about {s}, in one sentence.",
]

ELEMENTS = (
    "Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium "
    "Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium "
    "Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton "
    "Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver "
    "Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium "
    "Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium "
    "Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth "
    "Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium "
    "Americium Curium Berkelium Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium "
    "Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium "
    "Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson"
).split()


def open_prompts():
    return [{"id": f"open-{i:03d}", "prompt": t.format(s=s)}
            for i, (s, t) in enumerate((s, t) for s in SUBJECTS for t in TEMPLATES)]


def boundary_prompts():
    items = []
    for z, name in enumerate(ELEMENTS, start=1):
        items.append({"id": f"elem-num-{z:03d}", "answer": str(z),
                      "prompt": f"What is the atomic number of {name}? Answer with the number only."})
        items.append({"id": f"elem-name-{z:03d}", "answer": name.lower(),
                      "prompt": f"Which chemical element has atomic number {z}? Answer with the name only."})
    rng = random.Random(20261029)
    for i in range(200):
        a, b = rng.randint(101, 999), rng.randint(11, 99)
        items.append({"id": f"mul-{i:03d}", "answer": str(a * b),
                      "prompt": f"What is {a} × {b}? Answer with the number only."})
    return items


def token_probes():
    texts = [
        "Pick a random number between 1 and 100. Answer with the number only.",
        "Pick a random number between 1 and 10. Answer with the number only.",
        "Name a colour. Answer with one word.",
        "Name an animal. Answer with one word.",
        "Name a fruit. Answer with one word.",
        "Name a country. Answer with one word.",
        "Name a city. Answer with one word.",
        "Name a musical instrument. Answer with one word.",
        "Name a sport. Answer with one word.",
        "Name a vegetable. Answer with one word.",
        "Pick a letter of the alphabet. Answer with the letter only.",
        "Name a day of the week. Answer with one word.",
        "Name a month. Answer with one word.",
        "Name a planet. Answer with one word.",
        "Name a tree. Answer with one word.",
        "Name a flower. Answer with one word.",
        "Name a metal. Answer with one word.",
        "Name an emotion. Answer with one word.",
        "Name a programming language. Answer with one word.",
        "Name a bird. Answer with one word.",
        "Is the sky blue? Answer yes or no.",
        "Is water wet? Answer yes or no.",
        "Heads or tails? Answer with one word.",
        "Rock, paper or scissors? Answer with one word.",
        "Name a shape. Answer with one word.",
        "Name a famous scientist. Answer with the surname only.",
        "Name a famous painter. Answer with the surname only.",
        "Name a breakfast food. Answer with one word.",
        "Name a drink. Answer with one word.",
        "Name a piece of furniture. Answer with one word.",
        "Name a body part. Answer with one word.",
        "Name a weather condition. Answer with one word.",
        "Name an ocean. Answer with one word.",
        "Name a mammal that lives in the sea. Answer with one word.",
        "Name a gemstone. Answer with one word.",
        "Name a board game. Answer with one word.",
        "Name a dance. Answer with one word.",
        "Name a spice. Answer with one word.",
        "Name a tool. Answer with one word.",
        "Name a vehicle. Answer with one word.",
    ]
    return [{"id": f"tok-{i:02d}", "prompt": t} for i, t in enumerate(texts)]
