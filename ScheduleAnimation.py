import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ligo.skymap.io.fits import read_sky_map
from astropy.io import ascii
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from ligo.skymap import plot


def full_sch_ani(result_path, skymap_path, ns_total, ew_total, save_gif = False, name = 'full.gif', figsize = (10, 8), decay = 4):

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='astro mollweide')
    prob, _ = read_sky_map(skymap_path)
    ax.imshow_hpx(prob, cmap = 'cylon')

    schedule = ascii.read(result_path)

    old_artists = []

    def plot_schedule(k, ax, schedule):
        for artist in old_artists:
            artist.remove()
        del old_artists[:]

        for i,row in enumerate(schedule[:k+1]):

            coord = SkyCoord(*[float(c) for c in row['coord'].split(',')], unit = 'deg')

            coords = SkyCoord(
                [ew_total, -ew_total, -ew_total, ew_total],
                [ns_total, ns_total, -ns_total, -ns_total],
                frame=coord.skyoffset_frame()
            ).icrs
            poly = plt.Polygon(
                np.column_stack((coords.ra.deg, coords.dec.deg)),
                alpha=np.exp((i-k)/decay),
                facecolor='green',
                edgecolor='black',
                transform=ax.get_transform('world'),
            )
            ax.add_patch(poly)
            old_artists.append(poly)

    anim = animation.FuncAnimation(fig, plot_schedule, frames=len(schedule), fargs=(ax, schedule))
    if save_gif: anim.save(name, writer=animation.PillowWriter())

    plt.show()

