# Exploring the evolutionary adaptations of the unique seahorse tail's muscle architecture through in-silico modelling and robotic prototyping

<h1>
  <a href="#"><img alt="overview" src="https://github.com/Co-Evolve/seahorse-muscle-architecture/blob/main/assets/cover.png?raw=true" width="100%"/></a>
</h1>

**Abstract:** Seahorses possess a unique tail muscle architecture that enables efficient grasping and anchoring onto objects. This prehensile ability is crucial for their survival, as it allows them to resist currents, cling to mates during reproduction, and remain camouflaged to avoid predators. Unlike in any other fish, the muscles of the seahorse tail form long, parallel sheets that can span up to eleven vertebral segments. This study investigates how this distinctive muscle arrangement influences the mechanics of prehension. Through in-silico simulations validated by a 3D-printed prototype, we reveal the complementary roles of these elongated muscles alongside shorter, intersegmental muscles. Furthermore, we show that muscles spanning more segments allow greater contractile forces and provide more efficient force-to-torque transmissions. Our findings confirm that the elongated muscle-tendon organization in the seahorse tail provides a functional advantage for grasping, offering insights into the evolutionary adaptations of this unique tail structure.

## Repository structure
* `seahorse_muscle_architecture/CAD`: contains a compressed folder (`CAD.zip`) with all SLDPRT (SolidWorks Part) files and STL exports of the components.
* `seahorse_muscle_architecture/silico`: contains the code used for the in-silico experiments.
* `seahorse_muscle_architecture/vivo`: contains the code for and resulting data of the real-world experiments.

## Installation
Create the [anaconda](https://www.anaconda.com/) environment using:

`conda env create -f environment.yml`

## Reproducing the experiments
The in-silico experiments can be reproduced using the following commands:

```bash
# navigate to the project root
conda activate seahorse-muscle-architecture
python -m seahorse_muscle_architecture.silico.experiments.{experiment name}
```

Replace `{experiment name}` with one of the following:
- `contraction_force_to_torque`: increases the contraction forces applied to two symmetrical HM beams over a five-second interval in a two-segment model, while simultaneously measuring the resulting torque on the vertebral joint. Outputs a plot that compares the resulting in-silico measurements with the corresponding real-world measurements. 
- `contraction_direction_to_torque`: varies the contraction direction of two contracting symmetrical HM beams in a two-segment model, and measures the resulting torque on the vertebral joint. Outputs a plot that compares the resulting in-silico measurements with the corresponding real-world measurements.
- `hm`: visualizes the ventral bending of an eleven-segment model resulting from the contraction of two symmetrical HM beams with varying segment spans. 
- `hm_lateral`: visualizes lateral bending of an eleven-segment model resulting from the contraction of a single HM beam with varying segment spans.
- `mvm`: visualizes the ventral bending of an eleven-segment model resulting from the contraction of a varying amount of the most distal MVM beams over a five-second interval.
- `underactuation`: visualizes the grasping of a cylindrical object by an eleven-segment model resulting from the contraction of two symmetrical HM beams spanning all eleven segments.

## Citation
```
@article{marzougui2025,
  title = {Exploring the Evolutionary Adaptations of the Unique Seahorse Tail\&\#x2019;s Muscle Architecture through {\textexclamdown}i{\textquestiondown}in Silico{\textexclamdown}/I{\textquestiondown} Modelling and Robotic Prototyping},
  author = {Marzougui, Dries and Das, Riddhi and Mazzolai, Barbara and Adriaens, Dominique and {wyffels}, Francis},
  year = {2025},
  journal = {Journal of The Royal Society Interface},
  volume = {22},
  number = {226},
  eprint = {https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2024.0876},
  pages = {20240876},
  doi = {10.1098/rsif.2024.0876},
}

```
