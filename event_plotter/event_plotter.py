"""
Plot bar charts of events that happen in particular time buckets.
"""
# Because I can't avoid calling matplotlib.use() before importing pyplot, stop
# pylint from complaining about the imports that follow.

# pylint: disable=wrong-import-position

from __future__ import print_function
from collections import Counter, defaultdict
from datetime import datetime
import sys

from dateutil.parser import parse as du_parse
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdate


def jlog(stt):
    "Log a string to stderr, with a timestamp."
    print("{}: {}".format(datetime.now().strftime("%H:%M:%S"), stt, file=sys.stderr))


UTC0 = datetime.utcfromtimestamp(0)
# Number of hours that axes can span.
AXIS_CHOICES = [1, 2, 3, 4, 6, 12, 24, 48, 168]
# Number of minutes' worth of events that buckets can contain.
BUCKET_CHOICES = [1, 2, 5, 10, 15, 20, 30, 60]

# Assuming a tab20 color scheme in Matplotlib, what order should we display
# colors in? Exclude the grey color.
BICOLOR_ORDER = [0, 2, 4, 6, 8, 10, 12, 16, 18]
NO_BICOLOR_ORDER = [0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19]
# This is the grey color for things outside the top N.
OTHER_COLOR = 14
TAB_20 = plt.get_cmap("tab20")

##############################################################################
class _EventBucket(object):
    """
    Track events that occur in time_bucket (a number of seconds since the epoch).
    """

    def __init__(self, time_bucket):
        self.time_bucket = time_bucket
        self.bucket_date = datetime.utcfromtimestamp(time_bucket)
        self.counts = dict()
        self.total = 0

    def __str__(self):
        tss = "Event bucket for {}: events {}, total count {}".format(
            self.bucket_date, sum(len(v) for k, v in self.counts.items()), self.total
        )
        return tss

    def __repr__(self):
        return "<{}>".format(self.__str__())

    def add(self, event, count=1):
        """
        Add an event (a string) to this bucket, with a given count (i.e.,
        weighting).
        """
        if event not in self.counts:
            self.counts[event] = list()
        self.counts[event].append(count)
        self.total += count

    def order(self, rank_by_total=False):
        """
        Return a list of (event name, tt) tuples, where tt is the number of
        events if rank_by_total is False, or the sum of each event count (i.e.,
        weighting) if rank_by_total is True.
        """
        return sorted(
            [
                (event, sum(count) if rank_by_total else len(count))
                for (event, count) in self.counts.items()
            ],
            key=lambda t: t[1],
            reverse=True,
        )


##############################################################################

##############################################################################
class _AxisHelper(object):
    """
    Collect events to be plotted on a single Matplotlib axis.
    """

    def __init__(self, axis_start_seconds, axis_end_seconds, bucket_seconds,
        rank_by_total=False):
        """
        axis_start_seconds & axis_end_seconds are epoch values over which the
        x-axis spans.

        bucket_seconds is the number of seconds per bucket (i.e., per bar to be
        displayed on this graph).
        """
        # The leftmost "seconds since the epoch" value on the x-axis.
        self.axis_start_seconds = axis_start_seconds
        self.axis_start_datetime = datetime.utcfromtimestamp(axis_start_seconds)
        # The rightmost "seconds since the epoch" value on the x-axis.
        self.axis_end_seconds = axis_end_seconds
        self.axis_end_datetime = datetime.utcfromtimestamp(axis_end_seconds)
        # The total number of seconds spanned by this axis.
        self.bucket_seconds = bucket_seconds
        self.rank_by_total = rank_by_total
        self.events = dict()
        for x_offset in range(
            (axis_end_seconds - axis_start_seconds) // self.bucket_seconds
        ):
            self.events[x_offset] = dict()
        self._rank_counts = defaultdict(lambda: 0)

    def __str__(self):
        tss = "Event axis from {} to {} with {} buckets, {} full, y size {}".format(
            self.axis_start_datetime,
            self.axis_end_datetime,
            len(self.events),
            len([x for x in self.events.values() if len(x) > 0]),
            self.y_size(),
        )
        return tss

    def __repr__(self):
        return "<{}>".format(self.__str__())

    def add_event(self, x_offset, event_obj):
        """
        Read the data from _EventBucket, and put it into the bucket at x_offset
        (i.e., whose x-axis offset is between 0 and N-1, where N is the total
        number of buckets to be displayed).
        """
        assert x_offset in self.events, "x offset {} is not in range {} to {}".format(
            x_offset, min(self.events), max(self.events)
        )
        for event, count in event_obj.counts.items():
            if event not in self.events[x_offset]:
                self.events[x_offset][event] = list()
            self.events[x_offset][event].extend(count)

    def y_size(self):
        "Return the highest total size of any bar on this graph."
        max_y_size = None
        rank_fn = sum if self.rank_by_total else len
        for edict in self.events.values():
            event_total = sum([rank_fn(count) for count in edict.values()])
            if max_y_size is None or event_total > max_y_size:
                max_y_size = event_total
        return max_y_size

    def tab20_color(self, top_n, bicolor, rank):
        """
        Return a color number from 0 to 19 in which to plot a bar for an event,
        depending on how often that event occurred (i.e., its rank).

        - 'rank': How often the event occurred, with 0 being the most common, 1
          next most common, and so on.

        - 'top_n': For events that are outside the top N, plot them in OTHER_COLOR.

        - 'bicolor': If True, then the most common event will alternate between
          two shades of the same color, the next most common event will
          alternate between shades of a different color, and so on. Ohterwise,
          the colors will follow NO_BICOLOR_ORDER.

        Returned is a color index, ranging from 0 to 19. That value can be
        divided by 20 and passed to the "tab20" color map from Matplotlib.
        """
        if rank >= top_n:
            rank = top_n
        if bicolor:
            self._rank_counts[rank] += 1
            use_color = OTHER_COLOR if rank == top_n else BICOLOR_ORDER[rank]
            return use_color + 1 if self._rank_counts[rank] % 2 == 0 else 0
        return OTHER_COLOR + 1 if rank >= top_n else NO_BICOLOR_ORDER[rank]

    def _initialize_color(self):
        self._rank_counts = defaultdict(lambda: 0)

    def _build_bar(self, edict, event_rank, top_n, bicolor):
        """
        Return an array of values between 0.0 and 0.95 which represent the
        values to be displayed in each pixel of the bar for this bucket. These
        values are applied by the tab20 colormap.
        """
        self._initialize_color()
        color_list = list()
        for event, count_list in sorted(edict.items(), key=lambda k: event_rank[k[0]]):
            rank = event_rank[event]
            if self.rank_by_total:
                for count in count_list:
                    use_color = self.tab20_color(top_n, bicolor, rank)
                    color_list.extend([use_color] * count)
            else:
                for count, cardinality in sorted(
                    Counter(count_list).items(), key=lambda k: k[1], reverse=True
                ):
                    use_color = self.tab20_color(top_n, bicolor, rank)
                    color_list.extend([use_color] * cardinality)
        return np.array(color_list) / 20.0

    def _x_formatter(self):
        "Intelligently display ticks on the x-axis."
        if self.axis_end_seconds - self.axis_start_seconds > 86400:
            return mdate.DateFormatter("%dT%H:%M")
        return mdate.DateFormatter("%H:%M")

    def _x_locator(self):
        "Intelligently locate the x-axis ticks."
        t_start_hour = self.axis_start_datetime.hour
        axis_hours = (self.axis_end_seconds - self.axis_start_seconds) // 3600
        if axis_hours >= 48:
            byhour = [
                x % 24 for x in range(t_start_hour, t_start_hour + axis_hours + 1, 2)
            ]
            return mdate.HourLocator(byhour=byhour)
        if axis_hours >= 12:
            byhour = [
                x % 24 for x in range(t_start_hour, t_start_hour + axis_hours + 1)
            ]
            return mdate.HourLocator(byhour=byhour)
        elif axis_hours == 6:
            return mdate.MinuteLocator(byminute=range(0, 60, 30))
        elif axis_hours == 3:
            return mdate.MinuteLocator(byminute=range(0, 60, 15))
        return mdate.MinuteLocator(byminute=range(0, 60, 5))

    def plot(
        self,
        axi,
        event_rank,
        top_n,
        y_size=None,
        bicolor=False,
        ylabel=None,
        title=None,
    ):
        """
        Given axi, a MatPlotLib Axes object, and event_rank, a map from event
        name to the common-ness of that event (where 0 is most common and the
        values increase from there), plot the bar graph for the events.

        Parameters are as defined in the EventPlotter plot() method.
        """
        if y_size is None:
            y_size = self.y_size()
        if y_size <= 0:
            y_size = 1
        if ylabel is None:
            ylabel = "Event count"
        if title is None:
            title = "Event counts"
        assert top_n >= 1 and (
            (bicolor and top_n <= 9) or (not bicolor and top_n <= 18)
        ), (
            "Must " "use top_n between 1 and {} when bicolor is {}, not top_n of {}"
        ).format(
            9 if bicolor else 18, bicolor, top_n
        )
        im_data = np.zeros((y_size, len(self.events), 4))
        for x_offset, edict in sorted(self.events.items()):
            y_offset = 0
            prev_color_num = None
            for color_num in self._build_bar(edict, event_rank, top_n, bicolor):
                if color_num != prev_color_num:
                    prev_color_num = color_num
                    use_color = TAB_20(color_num)
                im_data[y_offset, x_offset] = use_color
                y_offset += 1
        axi.xaxis_date()
        axi.imshow(
            im_data,
            origin="lower",
            aspect="auto",
            extent=[
                mdate.epoch2num(self.axis_start_seconds),
                mdate.epoch2num(self.axis_end_seconds),
                0,
                y_size,
            ],
        )
        axi.grid(which="both")
        axi.xaxis.set_major_formatter(self._x_formatter())
        axi.xaxis.set_major_locator(self._x_locator())
        axi.set_ylabel(ylabel)
        bucket_minutes = self.bucket_seconds // 60
        dfmt = "%Y-%m-%dT%H:%M:%S"
        axi.set_title(
            "{} every {} from {} to {} UTC".format(
                title,
                "{} minutes".format(bucket_minutes)
                if bucket_minutes != 1
                else "minute",
                self.axis_start_datetime.strftime(dfmt),
                self.axis_end_datetime.strftime(dfmt),
            )
        )


##############################################################################

##############################################################################
class EventPlotter(object):
    """
    Collect and plot events that happen at particular times.
    """

    def __init__(self):
        self._events = dict()
        # Keys are events, and values are how many times an event occurred.
        self.event_size = defaultdict(lambda: 0)
        # Number of minutes & seconds in an event bucket.
        self.bucket_minutes = self.bucket_seconds = None
        # Map from event to how common that event was.
        self.event_rank = dict()
        # Number of hours & seconds on the x-axis on a single plot.
        self.axis_hours = self.axis_seconds = None

    def add(self, dto, event, count=1):
        """
        Given

        - 'dto', a datetime object at which an event occurred.

        - 'event', a string representing the event.

        - 'count' (default 1), the "weight" of that event.

        Round the datetime to the nearest second and add that event & count to
        the bucket of events in that second.
        """
        assert isinstance(dto, datetime)
        # Granularity can't be less than 1 second.
        s_second = int((dto - UTC0).total_seconds())
        if s_second not in self._events:
            self._events[s_second] = _EventBucket(s_second)
        self._events[s_second].add(event, count)

    def _parse_axis(self, axis, all_ts):
        """
        Calculate the number of hours per axis, and store it in
        self.axis_hours.  Also convert that to seconds and store it in
        self.axis_seconds.
        """
        self.axis_hours = axis
        if axis == "auto" or axis is None:
            total_seconds = all_ts[-1] - all_ts[0]
            total_hours = total_seconds // 3600
            # Try to make there be about 3 axes.
            axis_count = np.ceil(total_hours / np.array(AXIS_CHOICES))
            best_index = np.nonzero(axis_count >= 3)[0]
            best_index = 0 if best_index.shape == (0,) else best_index[-1]
            self.axis_hours = AXIS_CHOICES[best_index]
        assert (
            self.axis_hours in AXIS_CHOICES
        ), "Axis setting of {} hours is invalid: choose from {}".format(
            self.axis_hours, AXIS_CHOICES
        )
        self.axis_seconds = self.axis_hours * 3600

    def _parse_bucket(self, bucket):
        """
        Calculate the number of minutes per bucket, and store it in
        self.bucket_minutes.  Also convert that to seconds and store it in
        self.bucket_seconds.
        """
        self.bucket_minutes = bucket
        # How many minutes should we group long queries together under?
        if bucket == "auto" or bucket is None:
            # We'd like about 100 bars per axis.
            bar_count = np.ceil(self.axis_hours * 60 / np.array(BUCKET_CHOICES))
            best_index = np.nonzero(bar_count >= 100)[0]
            best_index = 0 if best_index.shape == (0,) else best_index[-1]
            self.bucket_minutes = BUCKET_CHOICES[best_index]
        assert (
            self.bucket_minutes in BUCKET_CHOICES
        ), "Bucket setting of {} minutes is invalid: choose from {}".format(
            self.bucket_minutes, BUCKET_CHOICES
        )
        self.bucket_seconds = self.bucket_minutes * 60

    def _calculate_rank(self, rank_by_total):
        """
        Once all events have been added to this object, rank them from most
        common to least common.
        """
        self.event_size = defaultdict(lambda: 0)
        for event_obj in self._events.values():
            for (event, count) in event_obj.order(rank_by_total=rank_by_total):
                self.event_size[event] += count
        rank_num = 0
        self.event_rank = dict()
        for event, _ in sorted(
            self.event_size.items(), key=lambda k: k[1], reverse=True
        ):
            self.event_rank[event] = rank_num
            rank_num += 1

    @staticmethod
    def _tab20_color(top_n, bicolor, rank):
        if rank >= top_n:
            return OTHER_COLOR if bicolor else OTHER_COLOR + 1
        return BICOLOR_ORDER[rank] if bicolor else NO_BICOLOR_ORDER[rank]

    def _plot_legend(self, axi, top_n, bicolor):
        """
        On the bottom axes, show the legend of event totals, colors, and names.
        """
        x_start, y_start = 0.00, 0.95
        from matplotlib.patches import Rectangle

        patch_height = 0.05
        all_count = sum(self.event_size.values())
        for event, rank_num in sorted(self.event_rank.items(), key=lambda k: k[1]):
            if rank_num == top_n:
                qtext = "(all other events)"
                qcount = all_count
            else:
                qtext = event
                qcount = self.event_size[event]
                all_count -= qcount
            use_color = self._tab20_color(top_n, bicolor, rank_num) / 20.0
            axi.text(
                x_start + 0.01,
                y_start,
                qcount,
                ha="right",
                va="center",
                fontsize="large",
            )
            axi.add_patch(
                Rectangle(
                    (x_start + 0.020, y_start - patch_height / 2),
                    0.01,
                    patch_height,
                    color=TAB_20(use_color),
                )
            )
            legend_offset = 0.035
            if bicolor:
                axi.add_patch(
                    Rectangle(
                        (x_start + 0.035, y_start - patch_height / 2),
                        0.01,
                        patch_height,
                        color=TAB_20(use_color + 0.05),
                    )
                )
                legend_offset += 0.015
            axi.text(
                x_start + legend_offset, y_start, qtext, va="center", fontsize="large"
            )
            y_start -= 0.1
            if rank_num >= top_n:
                break
            # Use a two-column layout in non-bicolor mode.
            if not bicolor and rank_num == 9:
                x_start, y_start = 0.5, 0.95
        axi.axis("off")

    def plot(
        self,
        axis="auto",
        bucket="auto",
        rank_by_total=False,
        top_n=None,
        bicolor=False,
        equal_y_axes=False,
        ylabel=None,
        title=None,
        suptitle=None,
        return_axes=False,
    ):
        """
        After you've added all events to this object, bucketize them and plot
        them on one or more sets of axes, with a legend. Return the resulting
        MatPlotLib figure object.

        Parameters:

        - axis (default "auto"): Several graphs will be plotted on the same
          figure.  What time range, in hours, should the x-axis for each graph
          span? Use "auto" to have the value automatically chosen.

        - bucket (default "auto"): How many minutes wide should each bucket of
          events be? The timestamps of the events will be grouped into buckets
          of this width. Use "auto" to have the value automatically chosen.

        - rank_by_total (default False): For each event, there's a count
          associated with it. For example:

          Event1: [10, 2]
          Event2: [2, 1, 1, 1]
          Event3: [3, 2, 2]

          There were two of Event1, with counts 10 & 2, four of Event2 with
          counts 2 1 1 1, and three of Event3.

          If rank_by_total is False, then it's the number of events which
          determines the sort order. In the above example, the sort order is
          thus Event2, Event3, Event1 (because there are 4, 3, and 2 of them).

          If rank_by_total is True, then it's the sum of the counts which
          determines sort order. In the above example, the sort order would be
          Event1, Event3, Event2 (because the sum of the counts is 12, 7, and
          5).

        - bicolor (default False): In some circumstances (for example, when
          plotting long query durations), you want each count in the same event
          to be plotted with alternating dark & light colors from the "tab20"
          colormap. So, with Event1, above, the "10" would be plotted in dark
          blue and the "2" in the light blue. Set bicolor to True, if this is
          the behavior you want.

          Otherwise, with bicolor False, all counts for an event are plotted
          using the same color.

        - top_n (default None): How many of the top N events (calculated
          depending on how you set rank_by_total) should be displayed
          separately? For bicolor True, this must be a value between 1 and 9.
          For bicolor False, it must be between 1 and 18. None meant to use the
          maximum allowed value (9 or 18).

        - equal_y_axes (default False): For each separate graph, should the
          y-axis have the same scale? Set equal_y_axes to True if so, or False
          otherwise.

        - ylabel (default None): How should each y-axis be labeled? Default is
          "Event count".

        - title (default None): How should each graph be titled? Default is
          "Event counts".

        - suptitle (default None): If not None, use this for fig.suptitle().

        - return_axes (default False): Return not just the figure object, but
          also the _AxisHelper objects? Useful for testing.
        """
        if top_n is None:
            top_n = 9 if bicolor else 18
        all_ts = sorted(self._events)
        self._parse_axis(axis, all_ts)
        self._parse_bucket(bucket)
        self._calculate_rank(rank_by_total=rank_by_total)
        axis_start_seconds = all_ts[0] - all_ts[0] % self.axis_seconds
        my_axes = dict()
        axis_num = 0
        jlog(
            "Axis hours: {}, bucket minutes: {}".format(
                self.axis_hours, self.bucket_minutes
            )
        )
        axis_now_seconds = axis_start_seconds
        while axis_now_seconds < all_ts[-1]:
            my_axes[axis_num] = _AxisHelper(
                axis_start_seconds=axis_now_seconds,
                axis_end_seconds=axis_now_seconds + self.axis_seconds,
                bucket_seconds=self.bucket_seconds,
                rank_by_total=rank_by_total,
            )
            axis_num += 1
            axis_now_seconds += self.axis_seconds
        axis_count = axis_num
        jlog("Number of axes: {}".format(axis_count))
        for tss in all_ts:
            axis_num = (tss - axis_start_seconds) // self.axis_seconds
            x_offset = tss % self.axis_seconds // self.bucket_seconds
            my_axes[axis_num].add_event(x_offset, self._events[tss])
        fig = plt.figure(1)
        fig.clf()
        fig.set_size_inches((20, max(9, 3 * (axis_count + 1))))
        use_y_size = (
            max([aobj.y_size() for aobj in my_axes.values()]) if equal_y_axes else None
        )
        for axis_num in sorted(my_axes):
            jlog(my_axes[axis_num])
            axi = fig.add_subplot(axis_count + 1, 1, axis_num + 1)
            my_axes[axis_num].plot(
                axi,
                self.event_rank,
                top_n,
                bicolor=bicolor,
                y_size=use_y_size,
                ylabel=ylabel,
                title=title,
            )
        axi = fig.add_subplot(axis_count + 1, 1, axis_count + 1)
        self._plot_legend(axi, top_n=top_n, bicolor=bicolor)
        if suptitle:
            fig.suptitle(suptitle, weight="bold")
        fig.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.98)
        return (fig, my_axes) if return_axes else fig


def main(use_args=None):
    "Wrapper for when we're called from the command line."

    from argparse import ArgumentParser
    import os

    parser = ArgumentParser(
        description="Given a file with lines containing an "
        "event timestamp, optionally a count, and a description of the event, "
        "plot a histogram summary of those events"
    )
    parser.add_argument("eventFile", nargs="+", help="File with input data")
    parser.add_argument(
        "-c",
        "--count",
        action="store_true",
        help="Does the input file also include a count of how many times an "
        "event occurred, immediately after the timestamp?",
    )
    parser.add_argument(
        "-a", "--axis", type=int, help="Range, in hours, for each x-axis to span"
    )
    parser.add_argument(
        "-b", "--bucket", type=int, help="Width, in minutes, for each histogram bucket"
    )
    parser.add_argument(
        "-r",
        "--rank-by-total",
        action="store_true",
        help="Rank events in the same histogram by the total",
    )
    parser.add_argument(
        "--bicolor",
        action="store_true",
        help="Plot a bicolor graph (where each event is displayed in alternating "
        "dark and light colors)? Works best with --count",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        help="How many of the top events should be showed separately?",
    )
    parser.add_argument(
        "-q",
        "--equal-y-axes",
        action="store_true",
        help="Make the range of every separate y-axis the same?",
    )
    parser.add_argument("-y", "--ylabel", help="Label for each y-axis")
    parser.add_argument("-t", "--title", help="Title for each graph")
    parser.add_argument("-s", "--suptitle", help="Super-title for all graphs")
    parser.add_argument(
        "-o",
        "--output-file",
        default="events.png",
        help='Output filename (default "events.png")',
    )

    opp = parser.parse_args(args=use_args)
    assert not opp.rank_by_total or opp.count, \
        "--rank-by-total is only meaningful with --count"

    self = EventPlotter()
    for fname in opp.eventFile:
        with open(fname) as fhh:
            for lnum, line in enumerate(fhh.readlines()):
                lsp = line.split()
                count = 1
                if opp.count:
                    assert (
                        len(lsp) >= 3
                    ), "File {} line {} has less than 3 fields".format(fname, lnum + 1)
                    assert lsp[1].isdigit(), (
                        "File {} line {} second column value '{}' isn't a count"
                    ).format(fname, lnum + 1, lsp[1])
                    count = int(lsp[1])
                    estr = " ".join(lsp[2:])
                else:
                    assert (
                        len(lsp) >= 2
                    ), "File {} line {} has less than 2 fields".format(fname, lnum + 1)
                    estr = " ".join(lsp[1:])
                dval = du_parse(lsp[0])
                self.add(dval, estr, count=count)
    fig = self.plot(
        axis=opp.axis,
        bucket=opp.bucket,
        rank_by_total=opp.rank_by_total,
        top_n=opp.top_n,
        bicolor=opp.bicolor,
        equal_y_axes=opp.equal_y_axes,
        ylabel=opp.ylabel,
        title=opp.title,
        suptitle=opp.suptitle,
    )
    ofile = os.path.realpath(os.path.expanduser(opp.output_file))
    jlog("Writing output to {}".format(ofile))
    fig.savefig(ofile)
    return fig


_ = main() if __name__ == "__main__" else None
