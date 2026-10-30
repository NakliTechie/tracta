"""The question set: short factual items near a small model's knowledge boundary, plus one-word probes.

The element questions are fixed; the products and the order of all items come from a per-user seed,
so no two users send the same list. The generator matches experiments/drift-calibration/probes.py,
which is what the published results were measured on.
"""
import random
import re

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

TOKEN_PROBES = [
    "Pick a random number between 1 and 100. Answer with the number only.",
    "Pick a random number between 1 and 10. Answer with the number only.",
    "Name a colour. Answer with one word.", "Name an animal. Answer with one word.",
    "Name a fruit. Answer with one word.", "Name a country. Answer with one word.",
    "Name a city. Answer with one word.", "Name a musical instrument. Answer with one word.",
    "Name a sport. Answer with one word.", "Name a vegetable. Answer with one word.",
    "Pick a letter of the alphabet. Answer with the letter only.", "Name a day of the week. Answer with one word.",
    "Name a month. Answer with one word.", "Name a planet. Answer with one word.",
    "Name a tree. Answer with one word.", "Name a flower. Answer with one word.",
    "Name a metal. Answer with one word.", "Name an emotion. Answer with one word.",
    "Name a programming language. Answer with one word.", "Name a bird. Answer with one word.",
    "Is the sky blue? Answer yes or no.", "Is water wet? Answer yes or no.",
    "Heads or tails? Answer with one word.", "Rock, paper or scissors? Answer with one word.",
    "Name a shape. Answer with one word.", "Name a famous scientist. Answer with the surname only.",
    "Name a famous painter. Answer with the surname only.", "Name a breakfast food. Answer with one word.",
    "Name a drink. Answer with one word.", "Name a piece of furniture. Answer with one word.",
    "Name a body part. Answer with one word.", "Name a weather condition. Answer with one word.",
    "Name an ocean. Answer with one word.", "Name a mammal that lives in the sea. Answer with one word.",
    "Name a gemstone. Answer with one word.", "Name a board game. Answer with one word.",
    "Name a dance. Answer with one word.", "Name a spice. Answer with one word.",
    "Name a tool. Answer with one word.", "Name a vehicle. Answer with one word.",
]


def boundary_items(seed):
    """436 items: atomic numbers both ways, and 200 seeded three-digit x two-digit products, in seeded order."""
    rng = random.Random(seed)
    items = []
    for z, name in enumerate(ELEMENTS, start=1):
        items.append({"id": f"elem-num-{z:03d}", "answer": str(z),
                      "prompt": f"What is the atomic number of {name}? Answer with the number only."})
        items.append({"id": f"elem-name-{z:03d}", "answer": name.lower(),
                      "prompt": f"Which chemical element has atomic number {z}? Answer with the name only."})
    for i in range(200):
        a, b = rng.randint(101, 999), rng.randint(11, 99)
        items.append({"id": f"mul-{i:03d}", "answer": str(a * b),
                      "prompt": f"What is {a} × {b}? Answer with the number only."})
    rng.shuffle(items)
    return items


def norm(text):
    """First number in the answer if there is one, else its first word, lower case."""
    t = text.strip().lower()
    m = re.search(r"-?\d[\d,]*", t)
    if m:
        return m.group(0).replace(",", "")
    w = re.findall(r"[a-z]+", t)
    return w[0] if w else t
