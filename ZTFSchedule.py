import astroplan
from astropy.coordinates import ICRS, SkyCoord, AltAz
from astropy import units as u
from astropy.utils.data import download_file
from astropy.table import Table, QTable, join, hstack
from astropy.time import Time
from astropy_healpix import *
import healpy as hp
import numpy as np
from ligo.skymap.io import read_sky_map
from FullLoc import schedule_event
from healpy.pixelfunc import ud_grade
from astropy.time import Time

def ZTF_Schedule(prob, start_time, end_time, p = [0, 0.0025, 0.005, 0.0075, 0.01, 0.02],
                 filttime=u.Quantity(60, u.s), b_max=6, slew_time=15 * u.s,
                 nfield=100, precal=True, time_limit_sales=500, time_limit_blocks=500, MIP_gap_blocks=None, debug=False,
                 time_gap=None, nside = 128, startWithG = True):

    if type(prob) == str:
        raw_prob, _ = read_sky_map(prob)
        prob = ud_grade(raw_prob, nside, power=-2)  # power is -2 to keep skymap normalized

    # Table 1 from Bellm et al. (2019)
    # http://adsabs.harvard.edu/abs/2019PASP..131a8002B
    ns_nchips = 4
    ew_nchips = 4
    ns_npix = 6144
    ew_npix = 6160
    plate_scale = 1.01 * u.arcsec
    ns_chip_gap = 0.205 * u.deg
    ew_chip_gap = 0.140 * u.deg

    rcid = np.arange(64)

    chipid, rc_in_chip_id = np.divmod(rcid, 4)
    ns_chip_index, ew_chip_index = np.divmod(chipid, ew_nchips)
    ns_rc_in_chip_index = np.where(rc_in_chip_id <= 1, 1, 0)
    ew_rc_in_chip_index = np.where((rc_in_chip_id == 0) | (rc_in_chip_id == 3), 0, 1)

    ew_offsets = ew_chip_gap * (ew_chip_index - (ew_nchips - 1) / 2) + ew_npix * plate_scale * (
                ew_chip_index - ew_nchips / 2) + 0.5 * ew_rc_in_chip_index * plate_scale * ew_npix
    ns_offsets = ns_chip_gap * (ns_chip_index - (ns_nchips - 1) / 2) + ns_npix * plate_scale * (
                ns_chip_index - ns_nchips / 2) + 0.5 * ns_rc_in_chip_index * plate_scale * ns_npix

    ew_ccd_corners = 0.5 * plate_scale * np.asarray([ew_npix, 0, 0, ew_npix])
    ns_ccd_corners = 0.5 * plate_scale * np.asarray([ns_npix, ns_npix, 0, 0])

    ew_vertices = ew_offsets[:, np.newaxis] + ew_ccd_corners[np.newaxis, :]
    ns_vertices = ns_offsets[:, np.newaxis] + ns_ccd_corners[np.newaxis, :]

    def get_footprint(center):
        # FIXME: SkyOffsetFrame doesn't seem to do broadcasting very well...
        dx, dy, x0, y0 = np.broadcast_arrays(ew_vertices.to_value(u.deg), ns_vertices.to_value(u.deg),
                                             center[..., np.newaxis, np.newaxis].icrs.ra.deg,
                                             center[..., np.newaxis, np.newaxis].icrs.dec.deg)
        return SkyCoord(dx * u.deg, dy * u.deg, frame=SkyCoord(x0 * u.deg, y0 * u.deg).skyoffset_frame()).icrs

    # Get the ZTF field grid
    url = 'https://github.com/ZwickyTransientFacility/ztf_information/raw/master/field_grid/ZTF_Fields.txt'
    filename = download_file(url)
    field_grid = QTable(np.recfromtxt(filename, comments='%', usecols=range(3), names=['field_id', 'ra', 'dec']))
    field_grid['coord'] = SkyCoord(field_grid.columns.pop('ra') * u.deg, field_grid.columns.pop('dec') * u.deg)

    #Define hpx grid
    hpx = HEALPix(nside=npix_to_nside(len(prob)), frame=ICRS())

    # Define observing plan parameters
    # Exposure time
    exptime = 300 * u.second

    #Return time
    rettime = 30 * u.minute

    # Observing constraints
    observing_constraints = [
        astroplan.AirmassConstraint(2.5)
    ]

    # From https://doi.org/10.1088/1538-3873/aaecbe
    slew_speed = 2.5 * u.deg / u.s
    slew_accel = 0.4 * u.deg / u.s ** 2
    readout = 8.2 * u.s

    site = 'Palomar'

    # Get hpix footprints
    footprints = np.moveaxis(get_footprint(field_grid['coord']).cartesian.xyz.value, 0, -1)
    footprints_healpix = [np.unique(
        np.concatenate([np.asarray([], dtype=np.intp)] +
                       [hp.query_polygon(hpx.nside, v, nest=(hpx.order == 'nested')) for v in footprint]))
        for footprint in footprints]



    result = schedule_event(prob, start_time, end_time, exptime, field_grid, footprints_healpix, slew_speed, slew_accel,
                            filttime, site, observing_constraints, p=p, b_max=b_max, slew_time=slew_time,
                            nfield=nfield, time_limit_sales=time_limit_sales, time_limit_blocks=time_limit_blocks,
                            MIP_gap_blocks=MIP_gap_blocks, time_gap=time_gap)

    if 'filter_change' in result.keys():
        filtIsg = startWithG
        filt_list = []
        for i, row in enumerate(result):
            if row['filter_change']: filtIsg = not filtIsg
            filt_list.append('g' if filtIsg else 'r')

        result['filt'] = filt_list

    return result
