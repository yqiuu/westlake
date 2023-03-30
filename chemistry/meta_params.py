from dataclasses import dataclass


@dataclass
class MetaParameters:
    site_density: float = 1.5e15 # Site density on one grain [cm-2]
