
from bim4loc.maps import Map
from typing import Callable

def vanila_filter(m : Map,
                forward_existence_model : Callable,
                exist_solid_names : list[str], 
                notexist_solid_names : list[str]) -> None:
    for name in exist_solid_names:
        s = m.solids[m.solid_names.index(name)]
        belief = min(s.existance_belief +0.1, 1.0)
        s.set_existance_belief_and_shader(belief)
        
    for name in notexist_solid_names:
        s = m.solids[m.solid_names.index(name)]
        belief = max(s.existance_belief  - 0.1, 0.0)
        s.set_existance_belief_and_shader(belief)