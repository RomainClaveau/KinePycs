# KinePycs

<div style="text-align:center;"> <img src="KinePycs.png" /> </div>

<br />

**KinePycs** is an automated framework to study, solve and extract kinetics, either from simulations or experimental curves.

## Description

`KinePycs` is designed to be an all-round analysis tool for studying kinetics, whether stemming from simulations or experiments. The main idea behind is to work on the following differential equation:

$$ \frac{d \alpha}{dt}(t) = \mathcal{F}[t, \alpha(t)]$$

with $\alpha(t)$ being the function of interest and $\mathcal{F}$ a functional, depending on both $t$ and $\alpha(t)$, and acting as a source term. The overall kinetics is completely encoded through $\mathcal{F}$. As a result, being able to retrieve its mathematical expression amounts for characterizing the main processes behind. 

In particular, to ensure working with a bounded function, we may define $\lambda(t)$ such that

$$ \mathcal{F}[t, \alpha(t)] = \lambda(t) |\alpha(t)| $$

`KinePycs` allows for an autonomous search for $\lambda(t)$ mathematical expression through the functions space. Because a complete search in that space would lead to a combinatory explosion, a probabilistic approach through [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) (SA) is employed.

## Theoretical aspects

Soon

## Documentation

Soon

## Installation

Soon

## License

CC BY-NC-SA (https://creativecommons.org/licenses/by-nc-sa/4.0/)