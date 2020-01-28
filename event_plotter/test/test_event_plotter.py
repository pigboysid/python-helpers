"""
Tests for EventPlotter class and related classes & functions.
"""

from __future__ import print_function
from datetime import timedelta
import os
import re
from tempfile import NamedTemporaryFile

from dateutil.parser import parse as du_parser
import pytest

from event_plotter.event_plotter import EventPlotter, main as ep_main

##############################################################################
CAPTURE_DIRECTORY = os.path.realpath(
    os.path.join(os.path.split(__file__)[0], ".")
)


def cfile_path(fname):
    "Return a filename prepended with the capture directory full path."
    return os.path.join(CAPTURE_DIRECTORY, fname)


CFILE_EVENTS_ALL = cfile_path("events_all.txt")
CFILE_EVENTS_DETAIL = cfile_path("events_detail.txt")


##############################################################################
class TestEventPlotter(object):
    "Test auvik_ops.model.event_plotter classes and functions."

    @staticmethod
    @pytest.mark.parametrize("_event_cap", [CFILE_EVENTS_ALL, CFILE_EVENTS_DETAIL])
    def test_mainline_playback(_event_cap):
        """
        Plot events in the _event_cap file.

        This test doesn't carefully check results: it's mostly for code coverage.
        """
        ofile = NamedTemporaryFile(delete=False).name + ".png"
        eargs = []
        if _event_cap == CFILE_EVENTS_DETAIL:
            eargs += ["--count", "--bicolor", "--top-n", "8"]
        fig_obj = ep_main(use_args=[_event_cap, "-o", ofile] + eargs)
        assert fig_obj, "Expected a nonempty return value from main()"
        assert os.path.isfile(ofile), "Expected a file to be plotted"
        os.unlink(ofile)

    @staticmethod
    def test_repr_and_str():
        "Check the output of internal __repr__() methods, to improve coverage."
        # First argument to add() must be a datetime.
        with pytest.raises(AssertionError):
            EventPlotter().add("2019-12-01T00:00:00", "This is the start event")
        eplot = EventPlotter()
        eplot.add(du_parser("2019-11-30T00:00:00"), "This is the start event")
        eplot.add(du_parser("2019-11-30T00:00:00"), "This is another start event")
        eplot.add(du_parser("2019-12-05T23:59:59"), "This is the end event")
        fig_obj, axes = eplot.plot(
            axis=48, bucket=60, equal_y_axes=True, return_axes=True
        )
        # I have events over 6 days, and 48h on each axis. So, I expect 3
        # _AxisHelper objects.
        assert len(axes) == 3, "Expected 3 _AxisHelper objects"
        _repr = axes[0].__repr__()
        assert (
            _repr
            == "<Event axis from 2019-11-30 00:00:00 to 2019-12-02 00:00:00 with 48 buckets, 1 full, y size 2>"
        ), "_AxisHelper __repr__ not as expected"
        # pylint: disable=protected-access
        for index, event in eplot._events.items():
            assert isinstance(index, int), "Index of _events should be an integer"
            _repr = event.__repr__()
            assert re.match(
                r"<Event bucket for 2019-.*: events [12], total count [12]>", _repr
            ), "_EventBucket __repr__ not as expected"
        # There should be four Matplotlib axis objects, one for each of the
        # three 2-day ranges and one for the legend. Because equal_y_axes is
        # True, I expect the y-axes to have the height of the time period with
        # the most events, which is the first axis (it's got two events; the
        # second axis has zero and the third has one).
        assert all(
            [ax.get_ylim() == (0.0, 2.0) for ax in fig_obj.axes[:-1]]
        ), "Expected ylim() of plotted axes to be (0.0, 2.0)"

    @staticmethod
    def test_axis_ranges():
        """
        Ensure axis ranges are calculated and labeled correctly. This gives
        coverage on _AxisHelper._x_locator() method.
        """
        first_date = du_parser("2019-11-30T00:00:00")
        # The ahours values below match those in the _x_locator method. And,
        # the size of the buckets (in minutes) is in expect_buckets.
        for (ahours, expect_buckets) in zip([48, 12, 6, 3, 1], [120, 60, 30, 15, 5]):
            eplot = EventPlotter()
            eplot.add(first_date, "Thes is the start event")
            eplot.add(
                first_date + timedelta(hours=ahours, seconds=-1),
                "This is the end event",
            )
            # Plot events on a graph whose axis spans 'ahours'. That means
            # there'll be only one axis. In Matplotlib, the legend is done in
            # its own separate axis, so there should be two Matplotlib axis
            # objects.
            fig_obj = eplot.plot(axis=ahours)
            # Looks like I have to render the figure in order to get axis
            # labels filled in.
            fig_obj.savefig(NamedTemporaryFile().name + ".png")
            assert len(fig_obj.axes) == 2, "Expected two Matplotlib axes"
            xtlab = [xtl.get_text() for xtl in fig_obj.axes[0].get_xticklabels()]
            lab_c = ahours * 60 // expect_buckets + 1
            assert len(xtlab) == lab_c, "Expected {} label names".format(lab_c)
            # The labels are <dd>T<HH>:<MM> for 2 hour spacing, and <HH>:<MM>
            # otherwise.
            assert all(
                [len(xtl) == 8 if ahours == 48 else 5 for xtl in xtlab]
            ), "Label string length not as expected"

    @staticmethod
    def test_plot_parameters():
        """
        Set plot() parameters that aren't otherwise tested, and ensure the
        graphs reflect the settings.
        """
        eplot = EventPlotter()
        eplot.add(du_parser("2019-11-30T00:00:00"), "This is the start event")
        eplot.add(du_parser("2019-11-30T00:00:00"), "This is another start event")
        eplot.add(du_parser("2019-11-30T23:59:59"), "This is the end event")
        exp_suptitle = "Some suptitle"
        exp_ylabel = "Some y label"
        fig_obj = eplot.plot(
            ylabel=exp_ylabel, title="Some title", suptitle=exp_suptitle
        )
        have_ylabel = fig_obj.axes[0].get_ylabel()
        assert have_ylabel == exp_ylabel, "Expected ylabel {}, got {}".format(
            exp_ylabel, have_ylabel
        )
        exp_title = "Some title every 2 minutes"
        have_title = fig_obj.axes[0].get_title()
        assert have_title.startswith(
            exp_title
        ), "Expected title to start with {}, got {}".format(exp_title, have_title)
        # It appears the only way to get at the supertitle is accessing the
        # protected member of the figure.
        #
        # pylint: disable=protected-access
        have_suptitle = fig_obj._suptitle.get_text()
        assert have_suptitle == exp_suptitle, "Expected suptitle {}, got {}".format(
            exp_suptitle, have_suptitle
        )
