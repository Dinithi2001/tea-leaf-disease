"""
All access to the tea-leaf ontology goes through SPARQL queries in this module.

Disease knowledge (which conditions, leaf maturity stages and symptoms favour each
disease) is read from the ontology at runtime rather than hard-coded in Python:

  * Environmental, maturity and symptom support are inferred by evaluating the SWRL
    rules stored in the ontology with SPARQL: a rule fires for a leaf when every
    condition class in the rule body is matched by one of the leaf's facts.
  * Each symptom has its own symptomSupported rule, so a disease's symptom score is
    the share of its symptom rules that fire.
  * Weather thresholds (e.g. HighHumidity: minHumidity 80) are data property values
    on each condition individual (minHumidity, minTemperature, maxTemperature,
    minSunshineHours, minLeafWetnessHours). They are read with SPARQL and compared
    against the weather readings to decide which conditions hold.
  * Treatment recommendations are Treatment individuals linked to each disease by
    hasTreatment, carrying treatmentText and stepOrder.
  * Symptom display text comes from rdfs:label / rdfs:comment on each symptom class,
    and each disease's incubationPeriod explains why weather may not match symptoms.

Every symptom, condition and maturity stage is a class with one individual named
<Class>_Instance (e.g. YoungLeaf_Instance of YoungLeaf); leaves point at those.

The SWRL rules are evaluated with SPARQL rather than sync_reasoner_pellet(), since
Pellet's bundled Jena jars require a JDK newer than what's installed here.
"""
import uuid
from contextlib import contextmanager

from owlready2 import get_ontology, default_world, destroy_entity

ONTOLOGY_PATH = "ontology/research2.rdf"
BASE_IRI = "http://www.semanticweb.org/user/ontologies/2026/7/tea-leaf-diseases-detection/"

onto = get_ontology(ONTOLOGY_PATH).load()
# The ontology IRI has no trailing "/", so owlready2 would otherwise look entities up
# under "...detection#" instead of "...detection/" where they are defined.
ns = onto.get_namespace(BASE_IRI)

PREFIXES = """
PREFIX t:    <http://www.semanticweb.org/user/ontologies/2026/7/tea-leaf-diseases-detection/>
PREFIX swrl: <http://www.w3.org/2003/11/swrl#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
"""


def sparql(query, params=()):
    return list(default_world.sparql(PREFIXES + query, list(params)))


# ---------- Rule queries ----------

# Lists the condition classes in the body of every SWRL rule whose head concludes
# <head property>(leaf, disease), restricted to subclasses of <parent class>
# (e.g. EnvironmentalCondition or LeafMaturityStage).
RULE_CONDITIONS_QUERY = """
SELECT ?disease ?cls WHERE {
  ?rule rdf:type swrl:Imp ; swrl:head ?head ; swrl:body ?body .
  ?head rdf:first ?headAtom .
  ?headAtom swrl:propertyPredicate t:%(head)s ; swrl:argument2 ?disease .
  ?body rdf:rest*/rdf:first ?atom .
  ?atom swrl:classPredicate ?cls .
  ?cls rdfs:subClassOf t:%(parent)s .
}
"""

# Evaluates those same SWRL rules against one leaf individual (the ?? parameter):
# a disease is supported when its rule has no body condition that the leaf fails
# to satisfy through <link property> (hasCondition / hasMaturity).
RULE_INFERENCE_QUERY = """
SELECT DISTINCT ?disease WHERE {
  ?rule rdf:type swrl:Imp ; swrl:head ?head ; swrl:body ?body .
  ?head rdf:first ?headAtom .
  ?headAtom swrl:propertyPredicate t:%(head)s ; swrl:argument2 ?disease .
  FILTER NOT EXISTS {
    ?body rdf:rest*/rdf:first ?atom .
    ?atom swrl:classPredicate ?cls .
    ?cls rdfs:subClassOf t:%(parent)s .
    FILTER NOT EXISTS { ?? t:%(link)s ?value . ?value rdf:type ?cls . }
  }
}
"""

# Like RULE_INFERENCE_QUERY, but returns the body condition classes of every rule that
# fires, so a disease with one rule per symptom can be scored by how many of its rules fire.
FIRED_RULE_CONDITIONS_QUERY = """
SELECT DISTINCT ?disease ?firedCls WHERE {
  ?rule rdf:type swrl:Imp ; swrl:head ?head ; swrl:body ?body .
  ?head rdf:first ?headAtom .
  ?headAtom swrl:propertyPredicate t:%(head)s ; swrl:argument2 ?disease .
  ?body rdf:rest*/rdf:first ?firedAtom .
  ?firedAtom swrl:classPredicate ?firedCls .
  ?firedCls rdfs:subClassOf t:%(parent)s .
  FILTER NOT EXISTS {
    ?body rdf:rest*/rdf:first ?atom .
    ?atom swrl:classPredicate ?cls .
    ?cls rdfs:subClassOf t:%(parent)s .
    FILTER NOT EXISTS { ?? t:%(link)s ?value . ?value rdf:type ?cls . }
  }
}
"""

# Display label and description of every symptom class.
SYMPTOM_INFO_QUERY = """
SELECT ?cls ?label ?comment WHERE {
  ?cls rdfs:subClassOf t:Symptom .
  OPTIONAL { ?cls rdfs:label ?label . }
  OPTIONAL { ?cls rdfs:comment ?comment . }
}
"""

# Time from infection to visible symptoms, for each disease that documents it.
INCUBATION_QUERY = """
SELECT ?disease ?period WHERE {
  ?disease t:incubationPeriod ?period .
}
"""

# Every leaf maturity stage class that has an individual a leaf can point at,
# with that individual's stageOrder (1 = youngest).
MATURITY_STAGES_QUERY = """
SELECT DISTINCT ?cls ?order WHERE {
  ?cls rdfs:subClassOf t:LeafMaturityStage .
  ?stage rdf:type ?cls .
  OPTIONAL { ?stage t:stageOrder ?order . }
}
"""

# Every numeric bound on every environmental condition individual, e.g.
# (HighHumidity_Instance, HighHumidity, minHumidity, 80.0).
THRESHOLD_DEFINITIONS_QUERY = """
SELECT ?condition ?cls ?prop ?bound WHERE {
  ?condition rdf:type ?cls .
  ?cls rdfs:subClassOf t:EnvironmentalCondition .
  ?condition ?prop ?bound .
  ?prop rdf:type owl:DatatypeProperty .
}
"""

# Which weather reading each bound property applies to, and how it is compared.
# min* bounds are inclusive and max* bounds exclusive, so a reading exactly on a shared
# boundary (HighTemperature min 25 / LowTemperature max 25) falls on one side only.
BOUND_PROPERTIES = {
    "minHumidity": ("humidityValue", ">="),
    "minTemperature": ("temperatureValue", ">="),
    "maxTemperature": ("temperatureValue", "<"),
    "minSunshineHours": ("sunshineHours", ">="),
    "minLeafWetnessHours": ("leafWetnessHours", ">="),
}
READING_LABELS = {
    "humidityValue": ("humidity", "%"),
    "temperatureValue": ("temperature", "°C"),
    "sunshineHours": ("sunshine", " h"),
    "leafWetnessHours": ("leaf wetness", " h"),
}
COMPARE = {">=": lambda value, bound: value >= bound, "<": lambda value, bound: value < bound}

# Treatment steps of every disease, in the order they should be shown.
TREATMENTS_QUERY = """
SELECT ?disease ?order ?text WHERE {
  ?disease t:hasTreatment ?treatment .
  ?treatment t:stepOrder ?order ; t:treatmentText ?text .
}
ORDER BY ?disease ?order
"""

ENVIRONMENT = {"head": "environmentallySupported", "parent": "EnvironmentalCondition", "link": "hasCondition"}
MATURITY = {"head": "maturitySupported", "parent": "LeafMaturityStage", "link": "hasMaturity"}
SYMPTOM = {"head": "symptomSupported", "parent": "Symptom", "link": "hasSymptom"}


def disease_name(disease_individual):
    """RedRust_Disease -> RedRust"""
    return disease_individual.name.removesuffix("_Disease")


def instance_of(class_name):
    """YoungLeaf -> the YoungLeaf_Instance individual"""
    return getattr(ns, f"{class_name}_Instance")


@contextmanager
def temp_leaf(**facts):
    """
    Creates a throwaway LeafInstance carrying the given property values
    (e.g. hasCondition=[...]), and removes it afterwards so facts from one
    diagnosis never leak into the next.
    """
    leaf = ns.LeafInstance(f"leaf_{uuid.uuid4().hex}", namespace=ns)
    for prop, values in facts.items():
        setattr(leaf, prop, list(values))
    try:
        yield leaf
    finally:
        destroy_entity(leaf)


def _threshold_rows():
    """(condition individual, condition class, bound property, bound value) for every known bound."""
    return [(condition, cls, prop, float(bound))
            for condition, cls, prop, bound in sparql(THRESHOLD_DEFINITIONS_QUERY)
            if prop.name in BOUND_PROPERTIES]


def evaluate_weather(readings):
    """
    readings: {"humidityValue": 96, "temperatureValue": 20, "sunshineHours": 2}
    Returns (condition individuals that hold, names of thresholds that were met).
    A condition holds when it has at least one bound in the ontology and the readings
    meet all of them; a bound on a reading that isn't available is never met.
    """
    bounds_per_condition = {}
    met = set()
    for condition, cls, prop, bound in _threshold_rows():
        reading, op = BOUND_PROPERTIES[prop.name]
        value = readings.get(reading)
        ok = value is not None and COMPARE[op](float(value), bound)
        bounds_per_condition.setdefault(condition, []).append(ok)
        if ok:
            met.add(f"{cls.name}.{prop.name}")
    conditions = [c for c, results in bounds_per_condition.items() if all(results)]
    return conditions, met


def get_condition_thresholds():
    """
    {"HeavyMist": [{"threshold": "HeavyMist.minHumidity", "reading": "humidityValue",
                    "label": "humidity", "unit": "%", "op": ">=", "bound": 95.0}, ...], ...}
    """
    thresholds = {}
    for _, cls, prop, bound in _threshold_rows():
        reading, op = BOUND_PROPERTIES[prop.name]
        label, unit = READING_LABELS[reading]
        thresholds.setdefault(cls.name, []).append({
            "threshold": f"{cls.name}.{prop.name}", "reading": reading, "label": label,
            "unit": unit, "op": op, "bound": bound,
        })
    return thresholds


def get_rule_conditions(rule_kind):
    """{"RedRust_Disease": ["BrightSunshine", "HighTemperature"], ...}"""
    conditions = {}
    for disease, cls in sparql(RULE_CONDITIONS_QUERY % rule_kind):
        conditions.setdefault(disease.name, []).append(cls.name)
    return {d: sorted(c) for d, c in conditions.items()}


def infer_supported_diseases(leaf, rule_kind):
    """Disease keys (e.g. "BlisterBlight_Disease") whose SWRL rule fires for the leaf."""
    return sorted(d.name for (d,) in sparql(RULE_INFERENCE_QUERY % rule_kind, [leaf]))


def get_maturity_stages():
    """["YoungLeaf", "MatureLeaf"], ordered by stageOrder so the youngest stage comes first."""
    stages = []
    for cls, order in sparql(MATURITY_STAGES_QUERY):
        if not isinstance(order, int):
            raise ValueError(f"{cls.name}_Instance needs an integer stageOrder in the ontology, "
                             f"found {order!r}")
        stages.append((order, cls.name))
    return [name for _, name in sorted(stages)]


def get_disease_symptoms():
    """{"RedRust": ["ReddishHairySpot", "RustySpot", ...], ...}, from the symptom SWRL rules."""
    return {d.removesuffix("_Disease"): s for d, s in sorted(get_rule_conditions(SYMPTOM).items())}


def get_symptom_info():
    """{"RustySpot": ("Rusty spot", "Olive-green to red circular spots ..."), ...}"""
    return {cls.name: (str(label) if label else cls.name, str(comment) if comment else "")
            for cls, label, comment in sparql(SYMPTOM_INFO_QUERY)}


def get_incubation_periods():
    """{"BlisterBlight": "8-10 days from penetration to the first visible symptom ...", ...}"""
    return {disease_name(disease): str(period) for disease, period in sparql(INCUBATION_QUERY)}


def get_treatments():
    """{"RedRust": ["Use rust-resistant plant varieties where available.", ...], ...}"""
    treatments = {}
    for disease, _, text in sparql(TREATMENTS_QUERY):
        treatments.setdefault(disease_name(disease), []).append(str(text))
    return treatments


def get_matched_symptoms(symptom_names):
    """{"RedRust": ["RustySpot"], ...}: the symptoms of each symptom SWRL rule that fires
    for a leaf carrying the reported symptoms."""
    observed = [instance_of(name) for name in symptom_names]
    with temp_leaf(hasSymptom=observed) as leaf:
        matched = {}
        for disease, cls in sparql(FIRED_RULE_CONDITIONS_QUERY % SYMPTOM, [leaf]):
            matched.setdefault(disease_name(disease), []).append(cls.name)
    return matched
