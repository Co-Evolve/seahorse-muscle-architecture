# Tendon Designer

The Tendon Designer is a browser app. With it, you put tendons (artificial muscles) on a model of the seahorse tail. Then you simulate the tail and measure how it bends.

You do not need to write code. You click holes in the plates of the tail to make a tendon. The app simulates the tail with the MuJoCo physics engine, and it shows the result immediately. You can save your design as a file and send it to your supervisor.

The app also has a **Help** panel. The buttons and panels can change between versions. This guide explains the concepts. For the exact buttons, use the in-app Help.

## Start the app

1. Open a terminal in the folder of the repository (`seahorse-muscle-architecture`).
2. Activate the Python environment. Use the command for your installation:
   - Conda: `conda activate seahorse-muscle-architecture`
   - venv: `source .venv/bin/activate`
3. Start the app:

   ```bash
   python -m seahorse_muscle_architecture.silico.tendon_designer.serve
   ```

4. The browser opens the app. If the browser does not open, copy the address from the terminal into the browser.
5. To stop the app, press `Ctrl+C` in the terminal.

The app runs only on your computer. It does not send data to the internet.

## Words that the app uses

| Word | Meaning |
|---|---|
| Segment | One ring of the tail. There are 11 segments, numbered 0 (at the body) to 10 (at the tip). Segment 0 does not move. |
| Vertebra | The bone at the center of a segment. Two vertebrae connect with a joint. |
| Joint | The connection between segment *i* − 1 and segment *i*. A joint turns about three axes: **pitch** (ventral or dorsal bending), **roll** (sideways bending) and **yaw** (twist along the tail). |
| Plate | One of the four bony plates around a segment. The plates are at the corners: ventral-dextral, ventral-sinistral, dorsal-sinistral and dorsal-dextral. The plates can glide a small distance. |
| Hole (tap) | A point in a plate where a tendon can go through. Each hole has a proximal side and a distal side. |
| Free point | A point that you put anywhere on a plate, if no hole is at the correct position. |
| Ventral / dorsal | Towards the belly / towards the back. |
| Sinistral / dextral | Towards the left side / towards the right side of the animal. |
| Proximal / distal | Towards the body / towards the tip of the tail. |

The axes of each segment are:

- **+x**: ventral (−x is dorsal).
- **+y**: sinistral.
- **−y**: dextral.
- **+z**: towards the tip of the tail.

The slice view shows one segment as you look along the tail. In this view, you see the four plates and their holes.

## Workflow

1. **Choose a preset.** A preset is a configuration that is ready to use. For example, "HM span 4 (paper)" is the hypaxial muscle (HM) pair from the paper. "Empty" has no tendons.
2. **Add a tendon.** Give the tendon a short name, for example `hm_left`. Names can only use letters, digits, `_` and `-`.
3. **Click the holes.** Click the start hole first. Then click the via holes in order. Then click the end hole. Use the slice view to go from segment to segment. A tendon can go towards the tip or towards the body.
4. **Split a tendon (optional).** A tendon can split into two distal ends, like a fork. Select a point of the tendon where the split starts. Then click the holes of the second branch. The two branches share one force.
5. **Add a free point (optional).** To put a point where there is no hole, hold **Shift** and click on a plate.
6. **Mirror the tendon (optional).** The mirror function makes a copy of the tendon on the other side (sinistral ↔ dextral).
7. **Simulate.** Move the activation slider of a tendon from 0 to 1. The tail bends in the 3D view.
8. **Read the results.** Read the bending angles, the vertebral torques and the tendon forces in the panels and plots.
9. **Run experiments and compare.** An experiment applies the same activation pattern to different configurations or tendon groups. The app shows the results next to each other.
10. **Export the data.** Export the measurements as a CSV file. You can open a CSV file in Excel, R or Python.
11. **Save your design.** Save the configuration as a JSON file. Send this file to your supervisor. The file contains all the tendons, free points and parameters.

## Tendon types

| Type | Behavior |
|---|---|
| Motor | The tendon pulls with a force. Activation 1 gives the maximum force (for example 10 N). |
| Position | The tendon shortens to a target length. Activation 1 gives the maximum shortening (for example 26 %). |
| Passive | The tendon has no actuator. It acts like a spring when the tail stretches it. |

## What the measurements mean

| Measurement | Meaning |
|---|---|
| Ventral angle | The bending of a segment towards the belly, in degrees. A positive value is ventral bending. |
| Lateral angle | The bending of a segment to the side, in degrees. A positive value is bending towards dextral. |
| Tip angle | The angle of the last segment relative to segment 0. |
| Vertebral torque | The turning force that all the tendons apply together on one joint. The app shows it in N·mm, for each axis (pitch, roll, yaw). |
| Moment arm | How much the length of a tendon changes when a joint turns (in mm per radian). A large moment arm gives more torque from the same force. |
| Tendon force | The pulling force of the tendon, in N. |
| Excursion | How much the tendon became shorter, in mm. Excursion = rest length − current length. |
| Strain | The excursion divided by the rest length, in %. |
| Work | The energy that the tendon gave to the tail, in J. Work = force × shortening, added up over time. |

The torque of one tendon on one joint is: −(moment arm) × (tendon force).

## For the PI: reproduce the results in Python

The JSON file of the student is the full description of the model. The Python code builds the same MuJoCo model as the browser. The tests compare the two builders.

**Build a standalone MJCF folder** (XML and meshes, loads in plain MuJoCo):

```bash
python -m seahorse_muscle_architecture.silico.tendon_designer.build_mjcf student_config.json --out-dir out/ [--name X]
python -m mujoco.viewer --mjcf=out/X.xml
```

**Run a ramp-and-hold protocol** headless and write a CSV with the same measurements as the app (angles, vertebral torques, tendon force, length, excursion, strain and work):

```bash
python -m seahorse_muscle_architecture.silico.tendon_designer.run_protocol student_config.json \
    --out results.csv --groups dextral --ramp 1.0 --hold 1.0 [--level 1.0] [--per-tendon-torques]
```

**Use the API** in your own experiment scripts:

```python
import mujoco
from seahorse_muscle_architecture.silico.tendon_designer import config as td
from seahorse_muscle_architecture.silico.tendon_designer.run_protocol import TendonSimulation

cfg = td.load_config("student_config.json")
errors = td.validate_config(cfg, td.load_catalog())   # [] when valid
mjcf_model = td.build_model(cfg)                       # dm_control mjcf.RootElement
model = td.compile_model(mjcf_model)                   # mujoco.MjModel
data = mujoco.MjData(model)
td.set_activations(model, data, cfg, {"hm_dextral": 0.5})   # a in [0, 1] -> ctrl

sim = TendonSimulation(cfg)              # or: measurements as in the app
sim.set_activation("hm_dextral", 1.0)
sim.advance(1.0)
m = sim.measure()                        # dict, see DESIGN.md "Measurement"
```

Other functions in `config.py`:

- `apply_tendon_config(mjcf_model, cfg, catalog)` adds the tendons to a base morphology that you already have.
- `expand_tendon_sites(tendon, catalog, free_points)` gives the MuJoCo site and pulley list of one tendon.
- `activation_to_ctrl(model, tendon_cfg, a)` gives the actuator control value. A motor gets `−a`. A position actuator gets `length0 · (1 − a · max_strain)`.
- `make_environment(cfg)` gives a `SeahorseMJCEnvironment` (moojoco) with the arena cameras and lights. In this environment, all names have the prefix `<model name>/`. Set `"orientation": "hanging"` in the parameters to get the hanging tail of the existing experiments.

Note: the `N_force` sensor of a motor tendon gives the control value (−1 to 0), not newtons. The tendon force in newtons is `−actuator_force × gear`. `TendonSimulation` and the app use this formula.

The contract between the browser and Python (configuration format, site rules, parameters, measurements) is in `DESIGN.md`. To run the tests:

```bash
.venv/bin/python -m unittest seahorse_muscle_architecture.silico.tendon_designer.tests.test_config -v
```
