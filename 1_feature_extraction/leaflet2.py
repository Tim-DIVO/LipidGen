# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# lipyphilic --- lipyphilic.readthedocs.io
#
# Released under the GNU Public Licence, v2 or any higher version
#

"""Assign leaflets --- :mod:`lipyphilic.leaflets.assign_leaflets`
=================================================================

:Author: Paul Smith
:Year: 2021
:Copyright: GNU Public License v2

This module provides methods for assigning lipids to leaflets in a bilayer.

Assigning leaflets in planar bilayers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The class :class:`lipyphilic.leaflets.assign_leaflets.AssignLeaflets` assigns
each lipid to a leaflet based on the distance in *z* to the midpoint of
the bilayer. Lipids may be assigned to the upper leaflet (indicated by `1`),
the lower leaflet (`-1`) or the bilayer midplane (`0`).

Input
------

Required:
  - *universe* : an MDAnalysis Universe object
  - *lipid_sel* : atom selection for *all* lipids in the bilayer, including e.g. sterols

Options:
  - *midplane_sel* : atom selection for lipid that may occupy the midplane
  - *midplane_cutoff* : atoms within this distance from the midpoint are considered to be the midplane
  - *n_bins* : split the membrane into *n_bins \\* n_bins* patches, and calculate local membrane midpoints for each patch

Output
------

  - *leaflets* : leaflet to which each lipid is assigned at each frame

Leaflet data are returned in a :class:`numpy.ndarray`, where each row corresponds
to an individual lipid and each column corresponds to an individual frame, i.e.
leaflets[i, j] refers to the leaflet of lipid *i* at frame *j*. The results are
accessible via the `AssignLeaflets.leaflets` attribute.


Example usage of :class:`AssignLeaflets`
----------------------------------------

An MDAnalysis Universe must first be created before using AssignLeaflets::

  import MDAnalysis as mda
  import lipyphilic as lpp

  u = mda.Universe(tpr, trajectory)

If we have used the MARTINI forcefield to study a phospholipid/cholesterol mixture,
we can assign lipids and cholesterol to the upper and lower as follows::

  leaflets = lpp.AssignLeaflets(
    universe=u,
    lipid_sel="name GL1 GL2 ROH"
  )

We then select which frames of the trajectory to analyse (`None` will use every
frame) and choose to display a progress bar (`verbose=True`)::

  leaflets.run(
    start=None,
    stop=None,
    step=None,
    verbose=True
  )

The results are then available in the :attr:`leaflets.leaflets` attribute as a
:class:`numpy.ndarray`. Each row corresponds to an individual lipid and each column
to an individual frame, i.e `leaflets.leaflets[i, j]` contains the leaflet
membership of lipid *i* at frame *j*. Lipid *i*, at frame *j*, is in the upper
leaflet if `leaflets.leaflets[i, j]==1` and in the lower leaflet if
`leaflets.leaflets[i, j]==-1`.

Allowing lipids in the midplane
-------------------------------

The above example will assign every lipid (including sterols) to either the upper
or lower leaflet. To allow cholesterol to be in the midplane, we can provide
a :attr:`midplane_sel` and :attr:`midplane_cutoff` to :class:`AssignLeaflets`::

  leaflets = lpp.AssignLeaflets(
    universe=u,
    lipid_sel="name GL1 GL2 ROH",
    midplane_sel="resname CHOL and name ROH C2",
    midplane_cutoff=12.0
  )

A cholesterol molecule that has both its *ROH* and *C2* atoms within *12* Å of
membrane midpoint will be assigned to the midplane, i.e. for cholesterol *i*
at frame *j* that is in the midplane, `leaflets.leaflets[i, j]==0`.

Changing the resolution of the membrane grid
--------------------------------------------

The first two examples compute a global membrane midpoint based on all the atoms
of the lipids in the membrane. Lipids are then assigned a leaflet based on their distance
in :math:`z` to this midpoint. This is okay for planar bilayers, but can lead to incorrect
leaflet classification in membranes with undulations. If your bilayer has undulations,
`AssignLeaflets` can account for this by creating a grid in :math:`xy`
of your membrane, calculating the local membrane midpoint in each patch,
then assigning leaflet membership based on distance in :math:`z` to the local membrane
midpoint. This is done through use of `n_bins`::

  leaflets = lpp.AssignLeaflets(
    universe=u,
    lipid_sel="name GL1 GL2 ROH",
    midplane_sel="resname CHOL and name ROH C2",
    midplane_cutoff=12.0,
    n_bins=10
  )

In this example, the membrane will be split into a *10 x 10* grid and a lipid
assigned a leaflet based on the distance to the midpoint of the patch the lipid
is in.


Assigning leaflets in membranes with high curvature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your membrane is a vesicle or bilayer with *very* large undulations, such as in a
`buckled membrane <https://aip.scitation.org/doi/pdf/10.1063/1.4808077>`__,
:class:`lipyphilic.leaflets.assign_leaflets.AssignLeaflets` will assign lipids to the wrong
leaflet

The class :class:`AssignCurvedLeaflets` can be used in these
scenaries to assign each lipid to a leaflet using `MDAnalysis' Leaflet Finder
<https://docs.mdanalysis.org/1.0.0/documentation_pages/analysis/leaflet.html>`__.
Lipids may still be assigned to the upper/outer leaflet (indicated by `1`), the lower/inner leaflet
(`-1`) or the membrane midplane (`0`).

Input
------

Required:
  - *universe* : an MDAnalysis Universe object
  - *lipid_sel* : atom selection for *all* lipids in the bilayer, including e.g. sterols
  - lf_cutoff : distance cutoff below which two neighbouring atoms will be considered to be in the same leaflet.
Options:
  - *midplane_sel* : atom selection for lipid that may occupy the midplane
  - *midplane_cutoff* : atoms further than this distance from the either leaflet are considered to be the midplane
  - pbc : bool, specifying whether or not to take periodic boundaries into account

Output
------

  - *leaflets* : leaflet to which each lipid is assigned at each frame

Leaflet data are returned in a :class:`numpy.ndarray`, where each row corresponds
to an individual lipid and each column corresponds to an individual frame, i.e.
leaflets[i, j] refers to the leaflet of lipid *i* at frame *j*. The results are
accessible via the `AssignLeaflets.leaflets` attribute.

Example usage of :class:`AssignCurvedLeaflets`
----------------------------------------------

An MDAnalysis Universe must first be created before using AssignCurvedLeaflets::

  import MDAnalysis as mda
  from lipyphilic.leaflets.assign_leaflets import AssignLeaflets

  u = mda.Universe(tpr, trajectory)

If we have used the MARTINI forcefield to study a phospholipid/cholesterol mixture,
we can assign lipids and cholesterol to the upper and lower as follows::

  leaflets = AssignCurvedLeaflets(
    universe=u,
    lipid_sel="name GL1 GL2 ROH",
    lf_cutoff=12.0,
    midplane_sel="name ROH",
    midplane_cutoff=10.0
  )

We then select which frames of the trajectory to analyse (`None` will use every
frame) and choose to display a progress bar (`verbose=True`)::

  leaflets.run(
    start=None,
    stop=None,
    step=None,
    verbose=True
  )

This will first use `MDAnalysis' Leaflet Finder
<https://docs.mdanalysis.org/1.0.0/documentation_pages/analysis/leaflet.html>`__ to assign
all lipids, excluding those in :attr:`midplane_sel`, to either the upper or lower leaflet. The
:class:`LeafletFinder` will consider two lipids to be in the same leaflet if they have `GL1` or
`GL2` atoms within :math:`12` Å of one another. From this, we find the two largest leaflets,
then assign the remaining phospholipids to a leaflet based on whichever leaflet they are closest
to.

The phospholipids do not change leaflets throughtout the trajectory, only cholesterol --- as specified
with :attr:`midplane_sel` and :attr:`midplane_cutoff`. Thus, at each frame, each cholesterol is
assinged a leaflet based on it's minimum distance to the leaflet. In the above example, if a cholesterol
is within :math:`10` Å of one leaflet it is assigned to that leaflet. If it is within :math:`10` Å of
*neither* or *both* leaflets then it is assigned to the midplane.

The results are then available in the :attr:`leaflets.leaflets` attribute as a
:class:`numpy.ndarray`. Each row corresponds to an individual lipid and each column
to an individual frame, i.e `leaflets.leaflets[i, j]` contains the leaflet
membership of lipid *i* at frame *j*. Lipid *i*, at frame *j*, is in the upper
leaflet if `leaflets.leaflets[i, j]==1` and in the lower leaflet if
`leaflets.leaflets[i, j]==-1`.

The classes and their methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AssignLeaflets
    :members:

.. autoclass:: AssignCurvedLeaflets
    :members:
    :inherited-members:
    :exclude-members: run
"""

from MDAnalysis.analysis.base import AnalysisBase
import MDAnalysis.analysis.distances
import MDAnalysis.analysis.leaflet
import numpy as np
import scipy.stats
import warnings # If not already imported

__all__ = [
    "AssignLeaflets",
    "AssignCurvedLeaflets",
]


class AssignLeafletsBase(AnalysisBase):
    """Abstract base class for leaflet identification."""

    def __init__(self, universe, lipid_sel, midplane_sel=None, midplane_cutoff=None):
        super().__init__(universe.trajectory)

        self.u = universe
        self.membrane = self.u.select_atoms(lipid_sel, updating=False)

        if (midplane_sel is not None) ^ (midplane_cutoff is not None):
            _msg = (
                f"midplane_sel is '{midplane_sel}' and midplane_cutoff "
                f"is {midplane_cutoff}. To assign molecules to the midplane, "
                "midplane_sel must be provided and midplane_cutoff must be "
                "greater than 0.",
            )
            raise ValueError(_msg)

        if (midplane_cutoff is not None) and (midplane_cutoff <= 0):
            _msg = "To assign molecules to the midplane, midplane_cutoff must be greater than 0."
            raise ValueError(_msg)

        self.potential_midplane = self.u.select_atoms(midplane_sel, updating=False) if midplane_sel else None
        self.midplane_cutoff = midplane_cutoff if midplane_cutoff else 0.0

        if self.potential_midplane and ((self.potential_midplane - self.membrane.residues.atoms).n_atoms > 0):
            _msg = (
                "midplane_sel contains atoms that are not present in molecules selected "
                "in lipid_sel. lipid_sel must cover *all* residues in the membrane.",
            )
            raise ValueError(_msg)

        self.results.leaflets = None

    @property
    def leaflets(self):
        return self.results.leaflets

    def _assign_leaflets(self):
        """Assign lipids to the upper (1) or lower (-1) leaflet."""
        # pragma: no cover

    def _find_midplane(self):
        """Determine which residues are in the midplane"""
        # pragma: no cover

    def filter_leaflets(self, lipid_sel=None, start=None, stop=None, step=None):
        """Create a subset of the leaflets results array.

        Filter either by lipid species or by the trajectory frames, or both.

        Parameters
        ----------
        lipid_sel : str, optional
            MDAnalysis selection string that will be used to select a subset of lipids present
            in the leaflets results array. The default is `None`, in which case data for all lipids
            will be returned.
        start : int, optional
            Start frame for filtering. The default is `None`, in which case the first frame is used
            as the start.
        stop : int, optional
            Stop frame for filtering. The default is `None`, in which case the final frame is used
            as the stop.
        step : int, optional
            Number of frames to skip when filtering frames. The deafult is `None`, in which case
            all frames between `start` and `stop` are used.

        """

        lipid_sel = "all" if lipid_sel is None else lipid_sel
        lipids = self.membrane.residues.atoms.select_atoms(lipid_sel)
        keep_lipids = np.in1d(self.membrane.residues.resindices, lipids.residues.resindices)

        start, stop, step = self.u.trajectory.check_slice_indices(start, stop, step)
        frames = np.arange(start, stop, step)
        keep_frames = np.in1d(self.frames, frames)

        return self.leaflets[keep_lipids][:, keep_frames]


class AssignLeaflets(AssignLeafletsBase):
    """Assign lipids in a bilayer to the upper leaflet, lower leaflet, or midplane."""

    def __init__(self, universe, lipid_sel, midplane_sel=None, midplane_cutoff=None, n_bins=1):
        """Set up parameters for assigning lipids to a leaflet.

        Parameters
        ----------
        universe : Universe
            MDAnalysis Universe object
        lipid_sel : str
            Selection string for the lipids in a membrane. The selection
            should cover **all** residues in the membrane, including cholesterol.
        midplane_sel :  str, optional
            Selection string for residues that may be midplane. Any residues not
            in this selection will be assigned to a leaflet regardless of its
            proximity to the midplane.
            The default is `None`, in which case all lipids will be assigned to
            either the upper or lower leaflet.
        midplane_cutoff : float, optional
            Minimum distance in *z* an atom must be from the midplane to be assigned
            to a leaflet rather than the midplane. The default is `0`, in which case
            all lipids will be assigned to either the upper or lower leaflet. Must
            be non-negative.
        n_bins : int, optional
            Number of bins in *x* and *y* to use to create a grid of membrane patches.
            Local membrane midpoints are computed for each patch, and lipids assigned
            a leaflet based on the distance to their local membrane midpoint. The
            default is `1`, which is equivalent to computing a single global
            midpoint.

        Note
        ----

        Typically, :attr:`midplane_sel` should select only sterols. Other lipids have
        flip-flop rates that are currently unaccessible with MD simulations, and thus
        should always occupy either the upper or lower leaflet.
        """
        super().__init__(
            universe=universe,
            lipid_sel=lipid_sel,
            midplane_sel=midplane_sel,
            midplane_cutoff=midplane_cutoff,
        )

        if not np.allclose(self.u.dimensions[3:], 90.0):
            _msg = (
                "AssignLeaflets requires an orthorhombic box - triclinic systems are not supported. "
                "Please use :class:`lipyphilic.leaflets.AssignCurvedLeaflets` instead."
            )
            raise ValueError(_msg)

        self.n_bins = n_bins
        self.results.leaflets = None

    def _prepare(self):
        # Output array
        self.results.leaflets = np.full(
            (self.membrane.n_residues, self.n_frames),
            fill_value=0,
            dtype=np.int8,  # smallest sized `np.int` is 1 byte, still 8 times smaller than using `int`
        )

    def _single_frame(self):
        # Atoms must be wrapped before creating a lateral grid of the membrane
        self.membrane.wrap(inplace=True)

        # Find the midpoint of the bilayer as a function of (x,y), using
        # `n_bins` grid points in each dimensions
        # Use all atoms in the membrane to get better statistics
        if self.n_bins > 1:
            x_bins = np.linspace(0.0, self._ts.dimensions[0], self.n_bins + 1)
            y_bins = np.linspace(0.0, self._ts.dimensions[1], self.n_bins + 1)
        else:
            # scipy.stats.binned_statistics raises Value error if there is only one bin
            x_bins = [0.0, self._ts.dimensions[0] + 1, self._ts.dimensions[0] + 2]
            y_bins = [0.0, self._ts.dimensions[1] + 1, self._ts.dimensions[1] + 2]

        memb_midpoint_xy = scipy.stats.binned_statistic_2d(
            x=self.membrane.positions[:, 0],
            y=self.membrane.positions[:, 1],
            values=self.membrane.positions[:, 2],
            statistic="mean",
            bins=(x_bins, y_bins),
            expand_binnumbers=True,
        )

        # Assign leaflets
        self._assign_leaflets(memb_midpoint_xy)

        # if necessary, find midplane residues
        if (self.potential_midplane is not None) and self.midplane_cutoff > 0.0:
            self._find_midplane(memb_midpoint_xy=memb_midpoint_xy)

    def _assign_leaflets(self, memb_midpoint_xy):
        """Assign lipids to the upper (1) or lower (-1) leaflet.

        Parameters
        ----------
        memb_midpoint_xy : BinnedStatistic2dResult
            Membrane grid created with stats.binned_statistic_2d. Contains the midpoint of
            each membrane patch.
        """

        # x and y may have different bins
        x_bins = memb_midpoint_xy.x_edge
        y_bins = memb_midpoint_xy.y_edge

        # get the binnumbers for each lipid
        lipid_x_bins, lipid_y_bins = (
            scipy.stats.binned_statistic_2d(
                x=self.membrane.positions[:, 0],
                y=self.membrane.positions[:, 1],
                values=self.membrane.positions[:, 2],
                statistic="mean",
                bins=(x_bins, y_bins),
                expand_binnumbers=True,
            ).binnumber
            - 1
        )  # These were bin numbers, now bin indices

        upper_leaflet = self.membrane[
            self.membrane.positions[:, 2]
            > (
                memb_midpoint_xy.statistic[lipid_x_bins, lipid_y_bins]
            )  # we don't to consider midplane_cutoff here
        ]
        self.results.leaflets[
            np.in1d(self.membrane.residues.resindices, upper_leaflet.residues.resindices),
            self._frame_index,
        ] = 1

        lower_leaflet = self.membrane[
            self.membrane.positions[:, 2]
            < (
                memb_midpoint_xy.statistic[lipid_x_bins, lipid_y_bins]
            )  # we don't to consider midplane_cutoff here
        ]
        self.results.leaflets[
            np.in1d(self.membrane.residues.resindices, lower_leaflet.residues.resindices),
            self._frame_index,
        ] = -1

        return

    def _find_midplane(self, memb_midpoint_xy):  # lgtm [py/inheritance/signature-mismatch]
        """Determine which residues are in the midplane

        Parameters
        ----------
        memb_midpoint_xy : BinnedStatistic2dResult
            Membrane grid created with stats.binned_statistic_2d. Contains the midpoint of
            each membrane patch.
        """

        # Atoms must be wrapped before so we can assign lipids to grid patches
        self.potential_midplane.wrap(inplace=True)

        # x and y may have different bins
        x_bins = memb_midpoint_xy.x_edge
        y_bins = memb_midpoint_xy.y_edge

        midplane_x_bins, midplane_y_bins = (
            scipy.stats.binned_statistic_2d(
                x=self.potential_midplane.positions[:, 0],
                y=self.potential_midplane.positions[:, 1],
                values=self.potential_midplane.positions[:, 2],
                statistic="mean",
                bins=(x_bins, y_bins),
                expand_binnumbers=True,
            ).binnumber
            - 1
        )  # These were bin numbers, now bin indices

        # First assume they're all midplane
        # Then find residues that have at least one atom further than
        # `midplane_cutoff` from the local midplane
        midplane_mask = np.full(self.potential_midplane.n_residues, fill_value=True, dtype=bool)

        not_midplane = (
            np.abs(
                self.potential_midplane.positions[:, 2]
                - memb_midpoint_xy.statistic[midplane_x_bins, midplane_y_bins],
            )
            > self.midplane_cutoff
        )

        # These residues have at least one atom in `potential_midplane`
        # that is more the `midplane_cutoff` from the local midplane
        midplane_mask[
            np.in1d(
                self.potential_midplane.residues.resindices,
                self.potential_midplane[not_midplane].resindices,
            ),
        ] = False

        midplane_residues = self.potential_midplane.residues[midplane_mask]

        # Assign midplane
        self.results.leaflets[
            np.in1d(self.membrane.residues.resindices, midplane_residues.resindices),
            self._frame_index,
        ] = 0

        return


class AssignCurvedLeaflets(AssignLeafletsBase):
    """Assign lipids in a membrane to the upper leaflet, lower leaflet, or midplane."""

    def __init__(self, universe, lipid_sel, lf_cutoff=15, midplane_sel=None, midplane_cutoff=None, pbc=True):
        """Set up parameters for assigning lipids to a leaflet.

        Parameters
        ----------
        universe : Universe
            MDAnalysis Universe object
        lipid_sel : str
            Selection string for the lipids in a membrane. The selection
            should cover **all** residues in the membrane, including cholesterol.
        lf_cutoff : float, optional
            Cutoff to pass to `MDAnalysis.analysis.leaflet.LeafletFinder`. Lipids closer
            than this cutoff distance apart will be considered to be in the same leaflet.
            The default is 15.0
        midplane_sel :  str, optional
            Selection string for residues that may be midplane. Any residues not
            in this selection will be assigned to a leaflet at ever frame.
            The default is `None`, in which case no molecules will be considered to be
            in the midplane.
        midplane_cutoff : float, optional
            Lipids with atoms selected in `midplane_sel` that are within this distance of
            a leaflet will be to that leaflet. If a molecule is within this distance of
            *neither* or *both* leaflets, it will be assigned to the midplane. The default
            is `None`.
        pbc : bool, optional
            Take periodic boundary conditions into account. The default is `True`.

        Note
        ----

        Typically, :attr:`midplane_sel` should select only sterols. Other lipids have
        flip-flop rates that are currently unaccessible with MD simulations, and thus
        should always occupy either the upper or lower leaflet.
        """
        super().__init__(
            universe=universe,
            lipid_sel=lipid_sel,
            midplane_sel=midplane_sel,
            midplane_cutoff=midplane_cutoff,
        )

        self.lf_cutoff = lf_cutoff
        self._pbc = pbc

        self.upper = None
        self.lower = None

    def _prepare(self):
        # Output array
        self.results.leaflets = np.full(
            (self.membrane.n_residues, self.n_frames),
            fill_value=0,
            dtype=np.int8,  # smallest sized `np.int` is 1 byte, still 8 times smaller than using `int`
        )

        self._assign_leaflets()

    def _single_frame(self):
        # if necessary, find midplane residues
        if self.potential_midplane is not None:
            self._find_midplane()

    def _assign_leaflets(self):
        """Assign lipids to the upper (1) or lower (-1) leaflet."""

        # Assign non-translocating lipids to their leaflets
        static = (
            self.membrane - self.potential_midplane if self.potential_midplane is not None else self.membrane
        )
        static_sel = "index " + " ".join(static.indices.astype(str))

        leaflets_static = MDAnalysis.analysis.leaflet.LeafletFinder(
            universe=self.u,
            select=static_sel,
            cutoff=self.lf_cutoff,
            pbc=self._pbc,
        )

        # make sure the two largest leaflets are in the first two indices
        sorter = np.argsort([g.n_atoms for g in leaflets_static.groups_iter()])[::-1]
        atom_groups = np.array(leaflets_static.groups(), dtype=object)[sorter]

        # If each group has the same number of atoms the above line will
        # cause each atom to be listed, rather than each AtomGroup,
        # thus increasing the dimension of the array to 2
        if atom_groups.ndim == 2:
            atom_groups = np.sum(atom_groups, axis=1)

        # find the center of geometry of each atom group
        cogs = np.array([ag.center_of_geometry() for ag in atom_groups])

        # Use this to define upper and lower leaflets
        lower, upper = atom_groups[np.argsort(cogs[:2, 2])]

        # And assign remaining groups to a leaflet
        for ag in atom_groups[2:]:
            upper_dists = MDAnalysis.lib.distances.distance_array(
                ag.positions,
                upper.positions,
            )

            lower_dists = MDAnalysis.lib.distances.distance_array(
                ag.positions,
                lower.positions,
            )

            if np.min(upper_dists) < np.min(lower_dists):
                upper += ag
            else:
                lower += ag

        #upper = upper[np.argsort(upper.indices)]
        #lower = lower[np.argsort(lower.indices)]

        #upper_mask = np.in1d(self.membrane.residues.resindices, upper.residues.resindices)
        #lower_mask = np.in1d(self.membrane.residues.resindices, lower.residues.resindices)

        #self._upper = upper
        #self._lower = lower
        #self.results.leaflets[upper_mask] = 1
        #self.results.leaflets[lower_mask] = -1

        # --- START: ADDED/MODIFIED LINES FOR RADIAL CORRECTION OF INNER/OUTER ---

        vesicle_center = self.membrane.center_of_geometry() # No 'box' argument

        # Calculate average radial distance for the current 'upper' and 'lower' groups
        avg_rad_dist_current_upper = np.mean(np.linalg.norm(upper.positions - vesicle_center, axis=1))
        avg_rad_dist_current_lower = np.mean(np.linalg.norm(lower.positions - vesicle_center, axis=1))

        # LiPyphilic convention: 1 for outer, -1 for inner.
        # self._upper should store the ATOMS of the outer leaflet.
        # self._lower should store the ATOMS of the inner leaflet.

        if avg_rad_dist_current_upper < avg_rad_dist_current_lower:
            # The Z-based 'upper' is radially INNER. Swap them.
            self._upper = lower[np.argsort(lower.indices)] # Radially outer becomes self._upper
            self._lower = upper[np.argsort(upper.indices)] # Radially inner becomes self._lower
            warnings.warn("LiPyphilic: Initial Z-sorted 'upper' leaflet was radially inner. Swapped to ensure '1' means outer, '-1' means inner.")
        elif avg_rad_dist_current_upper > avg_rad_dist_current_lower:
            # The Z-based 'upper' is radially OUTER. Keep assignment.
            self._upper = upper[np.argsort(upper.indices)]
            self._lower = lower[np.argsort(lower.indices)]
        else:
            # Radii are equal or something unexpected. Keep original Z-sort and warn.
            warnings.warn(f"LiPyphilic: Radially equidistant leaflets (avg_rad_upper={avg_rad_dist_current_upper:.2f} Å, "
                        f"avg_rad_lower={avg_rad_dist_current_lower:.2f} Å). "
                        "Retaining original Z-based sorting. Check system if a clear vesicle is expected.")
            self._upper = upper[np.argsort(upper.indices)]
            self._lower = lower[np.argsort(lower.indices)]

        # Now, populate self.results.leaflets using the correctly (radially) identified leaflets
        # These assignments are for ALL frames because these are the "static" lipids.
        outer_mask = np.in1d(self.membrane.residues.resindices, self._upper.residues.resindices)
        inner_mask = np.in1d(self.membrane.residues.resindices, self._lower.residues.resindices)

        self.results.leaflets[outer_mask, :] = 1  # Outer leaflet is 1
        self.results.leaflets[inner_mask, :] = -1 # Inner leaflet is -1

        # The 'return' statement was already at the end of _assign_leaflets
        # --- END: ADDED/MODIFIED LINES FOR RADIAL CORRECTION OF INNER/OUTER ---

        return

    def _find_midplane(self):
        """Determine which residues are in the midplane.

        Note, the below method of assigning molecules to the midplane is about two times
        faster than selecting dynamic atoms around upper/lower leaflet atoms, e.g:

        potential_upper = potential_midplane.select_atoms("around 10 global group upper", upper=upper).residues
        """

        upper_pairs = MDAnalysis.lib.distances.capped_distance(
            self.potential_midplane.positions,
            self._upper.positions,
            max_cutoff=self.midplane_cutoff,
            return_distances=False,
        )

        lower_pairs = MDAnalysis.lib.distances.capped_distance(
            self.potential_midplane.positions,
            self._lower.positions,
            max_cutoff=self.midplane_cutoff,
            return_distances=False,
        )

        upper_resindices = self.potential_midplane[upper_pairs[:, 0]].resindices
        lower_resindices = self.potential_midplane[lower_pairs[:, 0]].resindices

        # In upper but not lower = Upper Leaflet
        # And in lower but not upper = Lower Leaflet
        # Those in both or neither = midplane
        in_upper = np.in1d(upper_resindices, lower_resindices, invert=True)
        add_to_upper = np.in1d(self.membrane.residues.resindices, upper_resindices[in_upper])
        self.results.leaflets[add_to_upper, self._frame_index] = 1

        in_lower = np.in1d(lower_resindices, upper_resindices, invert=True)
        add_to_lower = np.in1d(self.membrane.residues.resindices, lower_resindices[in_lower])
        self.results.leaflets[add_to_lower, self._frame_index] = -1




      
        # --- START OF SIMPLIFIED "NEW PART" FOR RESOLVING AMBIGUOUS (0) LIPIDS ---
        
        # Re-build res_to_row_idx_map here for clarity, or ensure it's available if built once per frame
        res_to_row_idx_map = {resid.resindex: i for i, resid in enumerate(self.membrane.residues)}
        
        # current_box, outer_ref_positions, inner_ref_positions are already defined if needed by original part,
        # but let's redefine for clarity and ensure they are up-to-date for this section.
        current_box = self.u.dimensions if self._pbc else None
        # self._upper and self._lower should have been validated already at the start of the function.
        outer_ref_positions = self._upper.positions
        inner_ref_positions = self._lower.positions


        # Helper function for creating a safe residue string for warnings
        def get_safe_res_info(r_obj): # Renamed input to r_obj to avoid conflict with res loop var
            try:
                resname = r_obj.resname
            except MDAnalysis.exceptions.NoDataError:
                resname = "[NoResname]"
            try:
                resid_val = r_obj.resid
            except MDAnalysis.exceptions.NoDataError:
                resid_val = "[NoResid]"
            return f"{resname}{resid_val}"

        # Iterate through residues that were part of self.potential_midplane
        for res in self.potential_midplane.residues: # These are unique Residue objects
            row_idx = res_to_row_idx_map.get(res.resindex)
            if row_idx is None:
                continue 

            res_info_str = get_safe_res_info(res)

            # Only process if it's still ambiguous (0) after the first pass
            if self.results.leaflets[row_idx, self._frame_index] == 0:
                
                # Get the head bead(s) for THIS residue from self.potential_midplane AtomGroup
                # self.potential_midplane itself is the collection of all relevant head beads.
                head_beads_of_this_res_group = self.potential_midplane.select_atoms(f"resid {res.resid}")
                
                if head_beads_of_this_res_group.n_atoms == 0:
                    warnings.warn(f"Ambiguous lipid {res_info_str} (row {row_idx}) found no head beads "
                                  f"via 'self.potential_midplane.select_atoms(\"resid {res.resid}\")' "
                                  f"at frame {self.u.trajectory.frame}. Remains unresolved (0).")
                    continue
                
                query_positions = head_beads_of_this_res_group.positions

                try:
                    # Direct full Euclidean distance calculation to reference leaflets
                    min_dist_to_outer = np.min(
                        MDAnalysis.lib.distances.distance_array(query_positions, outer_ref_positions, box=current_box)
                    )
                    min_dist_to_inner = np.min(
                        MDAnalysis.lib.distances.distance_array(query_positions, inner_ref_positions, box=current_box)
                    )

                    if min_dist_to_outer < min_dist_to_inner:
                        self.results.leaflets[row_idx, self._frame_index] = 1  # Assign to outer
                    elif min_dist_to_inner < min_dist_to_outer:
                        self.results.leaflets[row_idx, self._frame_index] = -1 # Assign to inner
                    else: # Equidistant or both np.inf (if very far)
                        if min_dist_to_outer == np.inf: # Check if truly far from both
                            warnings.warn(f"Ambiguous lipid {res_info_str} (row {row_idx}) is very far from both leaflets (using distance_array). Defaulting to outer.")
                        else: # Equidistant but not infinitely far
                            warnings.warn(f"Ambiguous lipid {res_info_str} (row {row_idx}) is equidistant to leaflets (using distance_array). Defaulting to outer.")
                        self.results.leaflets[row_idx, self._frame_index] = 1 # Default to outer
                except Exception as e:
                    warnings.warn(f"Error during distance_array calculation for {res_info_str}: {e}. Lipid {res_info_str} remains 0.")
        # --- END OF SIMPLIFIED "NEW PART" ---