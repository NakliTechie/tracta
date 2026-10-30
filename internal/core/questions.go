// Package core holds tracta's question set, statistics, comparison, local store and report.
package core

import (
	"fmt"
	"math/rand/v2"
	"regexp"
	"strings"
)

// Elements in atomic-number order.
var Elements = strings.Fields(`Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium
Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium
Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton
Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver
Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium
Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium
Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth
Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium
Americium Curium Berkelium Californium Einsteinium Fermium Mendelevium Nobelium Lawrencium
Rutherfordium Dubnium Seaborgium Bohrium Hassium Meitnerium Darmstadtium Roentgenium Copernicium
Nihonium Flerovium Moscovium Livermorium Tennessine Oganesson`)

// TokenProbes are one-word prompts for the fingerprint; only their first token is read.
var TokenProbes = []string{
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
}

// Item is one question with its correct answer.
type Item struct {
	ID, Prompt, Answer string
}

// Generator names the question generator stored with a watch, so a watch is never read by a
// generator that would produce a different list.
const Generator = "go-pcg-v1"

// BoundaryItems returns 436 items: atomic numbers both ways and 200 seeded three-digit x two-digit
// products, in seeded order.
func BoundaryItems(seed uint64) []Item {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	items := make([]Item, 0, 2*len(Elements)+200)
	for i, name := range Elements {
		z := i + 1
		items = append(items,
			Item{fmt.Sprintf("elem-num-%03d", z), fmt.Sprintf("What is the atomic number of %s? Answer with the number only.", name), fmt.Sprint(z)},
			Item{fmt.Sprintf("elem-name-%03d", z), fmt.Sprintf("Which chemical element has atomic number %d? Answer with the name only.", z), strings.ToLower(name)})
	}
	for i := 0; i < 200; i++ {
		a, b := 101+r.IntN(899), 11+r.IntN(89)
		items = append(items, Item{fmt.Sprintf("mul-%03d", i), fmt.Sprintf("What is %d × %d? Answer with the number only.", a, b), fmt.Sprint(a * b)})
	}
	r.Shuffle(len(items), func(i, j int) { items[i], items[j] = items[j], items[i] })
	return items
}

var (
	numberRe = regexp.MustCompile(`-?\d[\d,]*`)
	wordRe   = regexp.MustCompile(`[a-z]+`)
)

// Norm returns the first number in an answer if there is one, else its first word, lower case.
func Norm(text string) string {
	t := strings.ToLower(strings.TrimSpace(text))
	if m := numberRe.FindString(t); m != "" {
		return strings.ReplaceAll(m, ",", "")
	}
	if w := wordRe.FindString(t); w != "" {
		return w
	}
	return t
}
