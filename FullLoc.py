import pandas as pd
from astropy import units as u
import numpy as np
from tqdm.auto import tqdm
import datetime as dt
from gurobipy import GRB, Model
from astropy.table import vstack
from astropy.time import Time
from astropy.table import QTable
from ligo.skymap.io import read_sky_map
from healpy.pixelfunc import ud_grade
import astroplan

# Get the time it takes to slew a given distance. Let's hope I've done my physics 1 properly
def getSlewTime(dist, slew_speed, slew_accel):
    # Below this distance, telescope can't travel far enough to get up to top speed
    if not type(dist) == u.quantity.Quantity:#Sometimes the functions return the degrees as floats, this handles that
        dist = dist * u.deg
    if dist <= slew_speed ** 2 / slew_accel:
        return 2 * np.sqrt(dist / slew_accel)
    else:
        return 2 * slew_speed / slew_accel + (dist - slew_speed ** 2 / slew_accel) / slew_speed

#Solve the max weighted coverage with fixed slew time between any 2 points and assuming a certain number filter
#changes take place. This is an ignore slew time problem
def fixed_slew(prob, nfield, fields, footprints_healpix):
    print('calculating healpix footprints')
    # Calculate healpix field footprint inverses
    total_footprint_healpix = np.unique(np.concatenate(footprints_healpix))  # The list of all hpx we can seet at least once
    total_footprint_healpix_inverse = {ipix: key for key, ipix in enumerate(total_footprint_healpix)}
    # indexing system; if hpx not in here, we don't see it, and if it is, then we can look at
    # corresponding index in footprints_inverse to see what fields it is in (by index; to get
    # field numbers, see what field_id those indicies correspond to in fields)

    footprints_inverse = [[] for _ in range(len(total_footprint_healpix))]

    for j, ks in enumerate(footprints_healpix):
        for k in ks:
            footprints_inverse[total_footprint_healpix_inverse[k]].append(j)


    t1 = dt.datetime.now()

    m = Model('Fixed-slew telescope')
    # define k here for future use (just so some math notation I used prior works)
    k = nfield

    print('creating variables')
    # add variables, x & y are column vectors
    x = np.asarray([m.addVar(vtype = GRB.BINARY, name= f'x{i}') for i in range(len(footprints_healpix))])
    y = np.asarray([m.addVar(vtype = GRB.BINARY, name= f'y{i}') for i in tqdm(range(len(prob)))])

    print('making objectives')

    # set objective
    m.setObjective(np.dot(prob, y), GRB.MAXIMIZE)

    #Look at these comments in a latex
    print('Adding Constraints')
    # add constraint $\sum_i x_i \leq k$
    print('Setsize Constraint')
    m.addConstr(np.sum(x) <= k, 'setsize constr')

    print('Visibility Constraint')
    # add other constraint $\sum_{e_{j}\in S_{i}} x_i \geq y_j$
    for j in tqdm(range(len(prob))):
        if j not in total_footprint_healpix:
            m.addConstr(0 >= y[j])
        else:
            m.addConstr(np.sum(x[footprints_inverse[total_footprint_healpix_inverse[j]]]) >=y[j])

    # run the model!
    t2 = dt.datetime.now()
    print(f'Model setup completed in {t2 -t1}')

    m.optimize()

    o_list = [int(v.varName[1:]) for v in m.getVars()[:len(x)] if np.round(v.x) == 1]
    #Round to handle floating point error

    pruned_footprint = [footprints_healpix[i] for i in o_list]

    return fields[o_list], pruned_footprint

def make_blocks(prob, field_list, footprints_healpix, end_time, start_time, exptime, filt_time, p,
                b_max = 6, slew_time = 10 * u.s, blocksize = 7, time_limit = None, gap = None, time_gap = None):
    if time_gap == None:
        time_gap = np.ones(b_max - 1) * 0 * u.s
    assert len(time_gap) == b_max - 1

    t1 = dt.datetime.now()

    total_footprint_healpix = np.unique(
        np.concatenate(footprints_healpix))  # The list of all hpx we can see at least once
    total_footprint_healpix_inverse = {ipix: key for key, ipix in enumerate(total_footprint_healpix)}

    footprints_inverse = [[] for _ in range(len(total_footprint_healpix))]

    for j, ks in enumerate(footprints_healpix):
        for k in ks:
            footprints_inverse[total_footprint_healpix_inverse[k]].append(j)

    m = Model('Block-schedule telescope')

    if not time_limit == None:
        m.Params.timelimit = time_limit

    if not gap == None:
        m.Params.MIPGapAbs = gap

    #add variables, B a rank 2 tensor, and all others are column vectors
    B = np.asarray([[m.addVar(vtype = GRB.BINARY, name = f'B{j},{i}') for i in range(len(field_list))] for j in range(b_max)])
    y = np.asarray([m.addVar(vtype = GRB.BINARY, name= f'y{i}') for i in range(len(prob))])
    t_o = np.asarray([m.addVar(vtype = GRB.CONTINUOUS, name = f't_s{i}') for i in range(b_max)])
    U = np.asarray([m.addVar(vtype = GRB.BINARY, name = f'U{i}') for i in range(b_max)])

    #set objective
    #subtract p times sum over B to disincentivize looking at excess fields, only "good enough" fields observed
    #What is "good enough" determined by choice of p
    m.setObjective(np.dot(prob, y) - np.sum(B) * p, GRB.MAXIMIZE)

    #I know I repeat the for loop bounds sometimes, but b_max is always so small it doesn't matter for runtime
    #and it makes it logically easier to follow

    #Lower bound on size of used blocks
    for i in range(b_max):
        m.addConstr(np.sum(B[i]) >= blocksize * U[i])

    #If you look at at least 1 field in a block, said block is now used to make observations
    for i in range(b_max):
        for j in range(len(B[i])):
            m.addConstr(U[i] >= B[i, j])

    #All unused blocks at the end of the night
    for i in range(1, b_max):
        m.addConstr(U[i] <= U[i-1])

    #Have to look at a field that contains a healpix pixel to look at said healpix pixel
    for j in range(len(prob)):
        if j not in total_footprint_healpix:
            m.addConstr(0 >= y[j])
        else:
            m.addConstr(np.sum(B[:,footprints_inverse[total_footprint_healpix_inverse[j]]]) >=y[j])


    #restrictions on when blocks can start & stop to not observe things not visible
    for i in range(b_max):
        for j in range(len(B[i])):
            m.addConstr(t_o[i] + np.sum(B[i]) * 2 * (exptime + slew_time).to(u.s).value + filt_time.to(u.s).value <=
                        B[i,j] * (field_list['observability_end_time'][j]-start_time - exptime).to(u.s).value +
                        (1-B[i,j]) * ((end_time-start_time).to(u.s).value))
            m.addConstr(t_o[i] >= B[i,j] * (field_list['observability_start_time'][j]-start_time).to(u.s).value)

    #Can't start on a new block before the previous finishes
    for i in range(1, b_max):
        m.addConstr(t_o[i] >= t_o[i-1] + np.sum(B[i-1]) * 2 * (exptime + slew_time).to(u.s).value + filt_time.to(u.s).value + time_gap[i-1].to(u.s).value)



    #run the model!
    t2 = dt.datetime.now()
    print(f'Model setup completed in {t2-t1}')

    m.optimize()

    b_list = [[int(v.varName.split(',')[1]) for v in b_i if np.round(v.x) == 1] for b_i in B]
    #round to handle floating point

    blocks = [field_list[b] for b in b_list]

    t_list = [v.x for v in t_o]

    p_cover = m.ObjVal + p * np.sum([[np.round(v.x) for v in b_i] for b_i in B])

    return blocks, t_list, p_cover

# Plan 1 observation batch, here an observation batch is defined as a group of fields to observe alltogether once and then come back to after >30 min and observe in the same order in another filter
# For this instance, I just need it to do the travelling salesperson to order the fields most efficiently
def salesperson(field_list, slew_speed, slew_accel, time_limit=None):
    t1 = dt.datetime.now()

    m = Model()

    if not time_limit == None:
        m.Params.timelimit = time_limit

    # you visit everything in this case
    k = len(field_list)

    # Define large Ms for big-M strategy,
    M3 = 720
    M4 = 360
    M5 = 135
    M6 = 205
    M7 = 115

    print('Defining Variables')

    #O[i,j] is 1 if you look at the jth field as your ith observation
    O = np.asarray(
        [[m.addVar(vtype=GRB.BINARY, name=f'O{j},{i}') for i in range(len(field_list))] for j in range(k)])

    print('Precalculating slew times, this can take up to a few minutes for longer field lists')
    t_i = dt.datetime.now()
    dist_table = np.ones(
        (k, k))  # Distance from field_list['coord'][i] to field_list['coord'][j] in degrees
    for i in tqdm(range(k)):
        for j in range(k):
            dist_table[i, j] = field_list['coord'][i].separation(field_list['coord'][j]).deg

    time_table = np.ones(np.shape(dist_table))
    for i in range(len(dist_table)):
        for j in range(len(dist_table[0])):
            time_table[i, j] = getSlewTime(dist_table[i, j], slew_speed, slew_accel).to(u.s).value
            #Have to get the value so it's a float const which plays nice with gurobi
    print(f'Time table generated in {dt.datetime.now() - t_i}')

    t_c = np.asarray(
        [m.addVar(vtype=GRB.CONTINUOUS, name=f't_c{i}') for i in range(k)])# in seconds, slew time for this observation

    # set objective to minimize slew time
    print('Setting objective')
    m.setObjective(np.sum(t_c), GRB.MINIMIZE)

    print('Beginning to add constraints')

    # Can only look at 1 field at a time
    print('Only look at 1 field at a time')
    for i in range(k):
        m.addConstr(np.sum(O[i]) == 1)

    # Add in general slew speed constraints
    # Add constraint that t[i] is >=exptime+slewtime from O[i-1] to O[i]
    print('Constraint on time from slew speed')
    for i in range(k):
        # Use the precalculated times
        m.addConstr(t_c[i] >= np.dot(np.dot(time_table, O[i - 1]), O[i]))
        # m.addConstr(t_c[i] >= exptime.value + t_c[i-1])

    # Have to look at each field at least once
    print('Constraint that you have to look at each field at least once')
    for i in range(k):
        m.addConstr(np.sum(O[:, i]) >= 1)

    # run the model!
    print(f'Model setup complated in {dt.datetime.now() - t1}')
    m.optimize()
    t2 = dt.datetime.now()
    print(f'Model completed in {t2 - t1}')

    # This bit converts the output of the model into an easily-readible dataframe
    # For consistency, it might be better to use an astropy table, but no one ever sees this code but me, and this works
    ord_list = []
    # Identify which field you're looking at each round
    for v in m.getVars()[0: k * k]:
        if np.round(v.x) == 1:  # np.round since sometimes you can get floating point errors
            ord_list.append(int(v.varName.split(',')[1]))

    # Identify the time you end each observation
    t_list = []
    for v in m.getVars()[k * k:k * (k + 1)]:
        t_list.append(v.x)
    t_list = np.asarray(t_list)

    # And then here is when we just repeat the observations a 2nd time to go back with a second filter
    ord_list = np.concatenate((ord_list, ord_list))
    t_list = np.concatenate((t_list, t_list))

    df = pd.DataFrame({'field': ord_list, 'times': t_list})
    return df



#This is the full-sky scheduler! It takes in all the parameters listed below and first solves the MILP for a fixed slew time, before breaking that list
#of fields up into several blocks you observe once, switch filters, and repeat, optimizing slew time within each block
def schedule_contiguous_event(prob, start_time, end_time, exptime, fields, footprints_healpix, slew_speed, slew_accel, filttime,
                   site, constr,
                   p = [0, 0.0025, 0.005, 0.0075, 0.01, 0.02], b_max = 6, slew_time = 12 * u.s,
                   nfield = 100, time_limit_sales = 500, time_limit_blocks = 500, MIP_gap_blocks = None,
                   time_gap = None, time_start = None, nside = 128):

    # Observer site
    observer = astroplan.Observer.at_site(site)

    # Calculate observing constraints with astroplan
    times = start_time + np.linspace(0, 1, 10000) * (end_time - start_time)
    observability = astroplan.is_event_observable(constr, observer, fields['coord'], times)

    # Select only fields that are observable
    keep = np.any(observability, axis=1)
    fields = fields[keep]
    footprints_healpix = [footprint_healpix for i,footprint_healpix in enumerate(footprints_healpix) if keep[i]]
    #footprints_healpix isn't a numpy array, so we have to do it this way
    observability = observability[keep]

    # Select only fields that are observable for at least 2 exptimes between (to rule out asteroids)
    fields['observability_start_time'] = [np.min(times[observable]) for observable in observability]
    fields['observability_end_time'] = [np.max(times[observable]) for observable in observability]
    keep = (fields['observability_end_time'] - fields['observability_start_time'] >= 2 * exptime)
    fields = fields[keep]
    footprints_healpix = [footprint_healpix for i,footprint_healpix in enumerate(footprints_healpix) if keep[i]]

    if type(prob) == str:
        raw_prob, _ = read_sky_map(prob)
        prob = ud_grade(raw_prob, nside, power=-2)  # power is -2 to keep skymap normalized

    if time_gap == None:
        time_gap = np.ones(b_max - 1) * 0 * u.s
    assert len(time_gap) == b_max - 1

    if time_start == None:
        time_start = dt.datetime.now()

    #Prune to top nfield fields for processing time next step
    flist, pruned_footprint = fixed_slew(prob, nfield, fields, footprints_healpix)

    #Break 1 if the localization is not visible
    if len(flist) == 0:
        coord = ['not visible']
        prob_covered = [0]
        schedule_final = QTable([coord, prob_covered], names=('coord', 'prob covered'))
        return schedule_final

    #This is where you make the blocks and then salesperson within each one, as well as choosing which p to use
    if type(p) == float:
        p = [p]

    blocks_list = []
    t_list_list = []
    p_cover_list = []
    block_time = dt.datetime.now()
    for i, p_i in enumerate(p):
        tblocks, tt_list, p_cover = make_blocks(prob, flist, pruned_footprint, end_time, start_time, exptime, filttime, p_i, b_max, slew_time, time_limit=time_limit_blocks, gap = MIP_gap_blocks, time_gap = time_gap)
        blocks_list.append(tblocks)
        t_list_list.append(tt_list)
        p_cover_list.append(p_cover)
    block_time = dt.datetime.now()-block_time

    #ask user which p they want to go forward with scheduling, unless they only provide one, then use that
    if len(p) > 1:
        for i, p_i in enumerate(p):
            size = np.sum([len(block) for block in blocks_list[i]])
            print(f'for p={p_i}, schedule {i} is {2*size} fields long, for an approxmiate observation time of {2* size * exptime} and has a probability coverage of {p_cover_list[i]}')
        index = int(input('Enter the number of the schedule you want to continue scheduling: '))
        blocks = blocks_list[index]
        t_list = t_list_list[index]
    else:
        blocks = blocks_list[0]
        t_list = t_list_list[0]

    raw_schedule = [salesperson(block, slew_speed, slew_accel, time_limit_sales) for block in blocks if len(block) > 0]

    #Intermediate variables for block num column
    block_lengths = np.asarray([len(b) for b in blocks])
    block_breaks = np.asarray([2*np.sum(block_lengths[:i+1]) for i in range(len(block_lengths))])
    block_number = np.asarray([i for i in range(len(blocks))])

    #Break 2 if the localization is not visible
    if len(raw_schedule) == 0:
        coord = ['not visible']
        prob_covered = [0]
        schedule_final = QTable([coord, prob_covered], names = ('coord', 'prob covered'))
        return schedule_final

    #Put it all together to make 1 large schedule
    schedule_final = vstack([blocks[i][np.asarray(raw_schedule[i]['field'])] for i in range(len(raw_schedule))])

    #Make a couple more of the columns & clean up ones no longer needed
    block_change = [block_number[i < block_breaks][0] for i in range(len(schedule_final))]
    final_times = []
    filt_cng = []
    for i,r in enumerate(raw_schedule):
        time_list = [exptime.to(u.s).value + r['times'][0] + t_list[i]]
        filt_block = [False]
        for j in range(1, len(r['times'])):
            temp = time_list[j - 1] + exptime.to(u.s).value + r['times'][j]
            temp2 = False
            if j == len(r['times'])//2:
                temp += filttime.to(u.s).value #Consider filter-change time
                temp2 = True
            time_list.append(temp)
            filt_block.append(temp2)
        final_times += time_list
        filt_cng += filt_block
    schedule_final['time'] = final_times * u.s + start_time #End up with observation time in jd
    schedule_final['exposure'] = np.ones(np.shape(schedule_final['time'])) * exptime #right now everything is the same (provided) exposure time
    schedule_final['filter_change'] = filt_cng #Put in place so you know when to change filters, eventually will expand to better handle different filters

    #Which number block you are looking at
    schedule_final['block_num'] = block_change

    #This is the probability coverage of each field, greater than total prob coverage as fields overlap & hpx are
    #counted multiple times
    field_indices = np.asarray([np.where(fields['field_id'] == obs)[0][0] for obs in schedule_final['field_id']])
    field_prob = [np.sum(prob[footprints_healpix[field_index]]) for field_index in field_indices]
    schedule_final['field_prob'] = field_prob

    #And this is the additional probability coverage of each field, in other words, the sum of the probabilities
    #of the hpx you are looking at for the first time for each field.
    total_hpx = np.asarray([])
    field_new_prob = []
    for i in range(len(field_indices)):
        new_hpx = np.asarray([hpx for hpx in footprints_healpix[field_indices[i]] if hpx not in total_hpx])
        if len(new_hpx > 0):
            field_new_prob.append(np.sum(prob[new_hpx]))
            total_hpx = np.concatenate((total_hpx, new_hpx))
        else:
            field_new_prob.append(0)
    schedule_final['new_prob'] = field_new_prob

    #This is where we make sure that the blocks are non-overlapping. If a block has an average slew time greater
    #than the fixed assumption, it can run long, so we need to correct such an occasion
    #This first code block sees how much time we need to push blocks back by so they don't overlap
    array = []
    lim = []
    for i in range(1, len(schedule_final)):
        if schedule_final['block_num'][i] != schedule_final['block_num'][i-1]:
            array.append(schedule_final['time'][i] - schedule_final['time'][i - 1] - exptime)
            lim.append(getSlewTime(schedule_final['coord'][i].separation(schedule_final['coord'][i-1]).deg, slew_speed, slew_accel))
    time_gap = [-t.to(u.s).value + lim[i].to(u.s).value if t.to(u.s).value < 0 else 0 for i,t in enumerate(array)] #get only negative values & invert to get sufficient gap time

    #This section offsets the blocks by the amount calculated above
    for i,t in enumerate(schedule_final['time']):
        schedule_final['time'][i] = t + u.s * np.sum(time_gap[:schedule_final['block_num'][i]])

    #And this section makes sure that the offset doesn't make you try to observe a field when it is not observable
    #I have never seen this happen, but the logic to handle it is here
    all_observable = (schedule_final['observability_start_time'] <= schedule_final['time'] - exptime).all() and (
            schedule_final['time'] <= schedule_final['observability_end_time']).all()
    if all_observable:
        #If it is all still observable, congrats! You have a schedule
        print(f'Schedule completed in {dt.datetime.now() - time_start}')
        runtime = np.ones(len(schedule_final)) * (dt.datetime.now() - time_start)
        block_time = np.ones(len(schedule_final)) * block_time
        schedule_final['runtime'] = runtime
        schedule_final['block_runtime'] = block_time

        return schedule_final
    else:
        #If it is not all observable, we add in more time between the offending blocks & try again. 
        print('Encountered observability error! Rerunning schedule with greater time gaps between offending blocks')
        while len(time_gap) < b_max - 1:
            time_gap.append(0)
        time_gap = time_gap * u.s
        return schedule_event(prob, start_time, end_time, exptime, fields, footprints_healpix, slew_speed, slew_accel,
                   filttime, site, constr, p=p, b_max = b_max, slew_time = slew_time,
                   nfield = nfield, time_limit_sales = time_limit_sales,
                              time_limit_blocks = time_limit_blocks, MIP_gap_blocks = MIP_gap_blocks,
                   time_gap = time_gap, time_start = time_start)

def schedule_event(prob, start_time, end_time, exptime, fields,
                              footprints_healpix, slew_speed, slew_accel,
                              filttime,
                              site, constr,
                              p=[0, 0.0025, 0.005, 0.0075, 0.01, 0.02],
                              b_max=6, slew_time=12 * u.s,
                              nfield=100, time_limit_sales=500,
                              time_limit_blocks=500, MIP_gap_blocks=None,
                              time_gap=None, time_start=None, nside=128):

    start_time = Time(start_time.mjd, format = 'mjd')

    observer = astroplan.Observer.at_site(site)

    next_set = Time(observer.twilight_evening_astronomical(start_time, 'next').mjd, format = 'mjd')
    next_rise = Time(observer.twilight_morning_astronomical(start_time, 'next').mjd, format = 'mjd')

    if next_set - next_rise > 0 * u.s:
        start_bound = start_time
        end_bound = next_rise
    else:
        start_bound = next_set
        end_bound = Time(observer.twilight_morning_astronomical(start_time,
                                                          'next').mjd, format='mjd')
    t_list = [(start_bound, end_bound)]

    while(True):
        start_bound = Time(observer.twilight_evening_astronomical(end_bound, 'next').mjd, format='mjd')
        end_bound = Time(observer.twilight_morning_astronomical(start_bound, 'next').mjd, format='mjd')
        if start_bound - end_time > 0 * u.s:
            break
        elif end_bound - end_time > 0 * u.s:
            t_list.append((start_bound, end_time))
            break
        else:
            t_list.append((start_bound, end_bound))

    final_schedule = None

    for (t_start, t_end) in t_list:
        if final_schedule != None:
            keep = []
            for field in fields:
                if field['field_id'] in final_schedule['field_id']:
                    keep.append(False)
                else:
                    keep.append(True)
            keep = np.asarray(keep)
            fields = fields[keep]
            footprints_healpix = [footprint for i, footprint in enumerate(footprints_healpix) if keep[i]]
        temp_schedule = schedule_contiguous_event(prob, t_start, t_end, exptime, fields,
                              footprints_healpix, slew_speed, slew_accel,
                              filttime,
                              site, constr,
                              p=p,
                              b_max=b_max, slew_time=slew_time,
                              nfield=nfield, time_limit_sales=time_limit_sales,
                              time_limit_blocks=time_limit_blocks, MIP_gap_blocks=MIP_gap_blocks,
                              time_gap=time_gap, time_start=time_start, nside=nside)

        if 'field_id' in temp_schedule.keys():
            if final_schedule == None:
                final_schedule = temp_schedule
            else:
                final_schedule = vstack([final_schedule, temp_schedule])

    return final_schedule

