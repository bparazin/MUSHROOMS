# MUSHROOMS

This is MUSHROOMS (Milp-Using ScHeduleR Of sky lOcalization MapS), a quick & lightweight set of programs to schedule 
follow-up of LIGO alerts and other MMA sources which provide healpix sky localization maps.
This was originally designed for ZTF, hence the ZTFSchedule.py file, but with the right inputs, FullLoc.py is almost 
fully telescope agnostic. Future development plans include a command-line parser and better handling of filters & 
reobservations to make it fully telescope agnostic. 

## Required Packages:

 - [Astroplan](https://astroplan.readthedocs.io/en/latest/)
 - [Astropy](https://www.astropy.org/)
 - [Gurobipy](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_python.html)
 - [Healpy](https://healpy.readthedocs.io/en/latest/)
 - [Ligo.skymap](https://lscsoft.docs.ligo.org/ligo.skymap/)
 - [Matplotlib](https://matplotlib.org/)
 - [Numpy](https://numpy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [Tqdm](https://tqdm.github.io/)

### Some notes on the required packages:

 - Gurobipy, and by extension Gurobi, needs a license to function; free academic licenses are avaliable [here](https://www.gurobi.com/academia/academic-program-and-licenses/)
 - Ligo.skymap does not work on windows. Those using it are encouraged to use WSL or some other way of running Linux on their PC

# Function Documentation:

## FullLoc.py

### schedule_event

`schedule_event(prob, start_time, end_time, exptime, fields, footprints_healpix, slew_speed, slew_accel, filttime, site, constr,
p = [0, 0.0025, 0.005, 0.0075, 0.01, 0.02], b_max = 6, slew_time = 12 * u.s,
                   nfield = 100, time_limit_sales = 500, time_limit_blocks = 500, MIP_gap_blocks = None,
                   time_gap = None, time_start = None, nside = 128):`

This function produces a schedule in 3 steps, first it runs a max weighted coverage algorithm to identify the top
nfields fields to look by probability coverage at in step 2, this is done to reduce runtime in step 2. In step 2, the 
fields are broken up into b_max number of blocks for observations, where a single block is a set of fields you observe
twice, performing a filter change before repeating observations. Finally, a travelling salesperson algorithm (TSP) is used to
schedule the optimal field plan within each block. For more information about the math behind the algorithms, refer to [this](https://www.overleaf.com/read/ctcpwkvrfdcq)
overleaf document.

Returns an astropy.table.table.Table with which is the ordered schedule.

### Required parameters

- prob: string or numpy array, either representing the path to a fits file of the skymap or a healpix mapping of the skymap
- start_time: astropy.time.core.Time, representing the earliest start time of observations
- end_time: astropy.time.core.Time, representing the latest end time of observations
- exptime: astropy.units.quantity.Quantity, representing the exposure time of the telescope used
- fields: astropy.table.table.Table of fields to look at, with columns for skycoord of field center and field id
- footprints_healpix: list of numpy arrays, where the ith data array is a list of the healpix in the footprint of the ith field in fields
- slew_speed: astropy.units.quantity.Quantity, representing the maximum slew speed of the telescope
- slew_accel: astropy.units.quantity.Quantity, representing the maximum slew acceleration of the telescope
- filttime: astropy.units.quantity.Quantity, representing the amount of time it takes for a filter change to happen
- site: String representing the observing site, for use in Astroplan.Observer.at_site
- constr: Any astroplan observing constraints, used in conjunctio with site to find observability for each field

### Optional parameters

- p: float or list of floats, representing different pruning aggressiveness, the objective function of the model is such that it only considers looking at fields that have a probability coverage greater than p, allowing you to reduce runtime by only looking at "good" fields
- b_max: int, maximum number of filter changes to do in the schedule/ number of blocks to break the schedule up into.
- slew_time: astropy.units.quantity.Quantity, fixed slew time to use in intermediate scheduling steps before actual slew times can be calculated
- nfield: int, amount of fields to prune to in step 1 of the algorithm
- time_limit_sales: int, time limit in seconds for each TSP algorithm
- time_limit_blocks: int, time limit in seconds for the block division algorithm from step 2
- MIP_gap_blocks: float < 1, the percentage gap between the incumbent best solution of step 2 and the theoretical maximum where they are considered to be equal.
- time_gap: list of astropy.units.quantity.Quantity, representing the amount of additional time to schedule between each block.
- nside: the healpix nside to up/downgrade the healpix skymap to once it is read


### All other methods in FullLoc.py perform steps of the schedule_event function and should not be called

## ZTFSchedule.py

### ZTF_Schedule

`def ZTF_Schedule(prob, start_time, end_time, p = [0, 0.0025, 0.005, 0.0075, 0.01, 0.02],
                 filttime=u.Quantity(60, u.s), b_max=6, slew_time=15 * u.s,
                 nfield=100, precal=True, time_limit_sales=500, time_limit_blocks=500, MIP_gap_blocks=None, debug=False,
                 time_gap=None, nside = 128)`

This is just a specialized version of schedule_event, since the field grid, footprints, slew speed and acceleration are
all known for ZTF. It simply calculates those, and then calls schedule_event, so all parameters needed are the same as
described in schedule_event. It adds one more optional parameter however, and that is the filter change time, which
means the same as the non-optional filttime in schedule_event

## ScheduleAnimation.py

### full_sch_ani
`full_sch_ani(result_path, skymap_path, ns_total, ew_total, save_gif = False, name = 'full.gif', figsize = (10, 8), decay = 4)`

This is a function that takes a completed schedule and animates a gif of the schedule so it can be easily visualized

### Required parameters

- result_path: string, path to the completed schedule
- skymap_path: string, path to the fits file of the skymap the schedule was constructed for
- ns_total: angular height of the field in the north-south direction
- ew_total: angular width of the field in the east-west direction
  - At the moment, the scheduler assumes rectangular fields, a future development plan is to also allow for circular fields

### Optional parameters

- save_gif: boolean, whether you want to save the gif constructed as a file
- name: string, path to where you want to save it and name of file
- figsize: tuple, width & height of gif in inches
- decay: how fast/slow you want old fields to decay in animation, higher numbers means slower decay