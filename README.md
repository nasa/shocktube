# ShockTube 
This repository contains code needed to solve a 1D shocktube. The repository is stuructured as follows. 
This main page contains codes `Analytical-Lax-Wendroff.py` and `Analytical-Roe.py` for solving using Lax Wendroff method and Roe Method.
`settings.json` contains all the initial conditions, CFL number, etc for different types of shocktube problems. The codes are all designed to solve the sod shock problem but can be changed for other types of problems. 

List of Folders 
- [ausm](ausm/): AUSM Flux splitting scheme from 1991
- [ausm+](ausm+/): AUSM+ scheme from 1996
- [ausm_higher_order](ausm_higher_order/): AUSM+ Higher order scheme from 1996


# Tutorial

This file shows how to run each of the codes mentioned in this Readme file along with governing equations. 

[tutorial](https://colab.research.google.com/github/nasa/shocktube/blob/main/tutorial.ipynb)


# Contributors
Paht Juangphanich

Kenji Miki

# License
[NASA Open Source Agreement](https://opensource.org/licenses/NASA-1.3)
