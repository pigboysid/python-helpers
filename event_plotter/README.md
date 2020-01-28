# Plot a histogram of events

Given a text file with a timestamp at the start of each line and a message on the rest of the line, plot a histogram of the most commonly-occurring messages.

## Basic usage and output

Sample data is found in `test/events_all.txt` and `test/events_detail.txt`. For reasonable data (time ranges between a few minutes and a couple of weeks, messages with a reasonable amount of repetition in them), no arguments are necessary:
```
python event_plotter.py -o test/events_all.png test/events_all.txt
```
Default output is to `events.png` in current working directory. The `-o` option writes the file to the given path; in the above example, it looks like this:

![events_all.png](https://github.com/pigboysid/python-helpers/blob/master/event_plotter/test/events_all.png)

Some notes about this output:
* The source file, `test/events_all.txt`, contains 12481 lines of the following format:
```
2018-09-11T00:00:13 [W] TenantManagerSingletonActor/DISALLOW_DISABLED_TENANT_START
2018-09-11T00:00:17 [W] TenantManagerSingletonActor/DISALLOW_DISABLED_TENANT_START
2018-09-11T00:00:39 [W] TenantManagerSingletonActor/DISALLOW_DISABLED_TENANT_START
2018-09-11T00:00:42 [W] TenantManagerSingletonActor/DISALLOW_DISABLED_TENANT_START
2018-09-11T00:01:18 [W] TenantManagerSingletonActor/DISALLOW_DISABLED_TENANT_START
[...]
2018-09-11T19:03:29 [I] TenantManagerSingletonActor/START_REQUEST
2018-09-11T19:03:29 [I] TenantManagerSingletonActor/TENANT_GUARDIAN_TERMINATED
2018-09-11T19:03:29 [I] TenantManagerSingletonActor/BACKEND_ACCEPTING_TENANTS
2018-09-11T19:03:29 [I] TenantManagerSingletonActor/NON_ROOT_TENANT_STOPPED
2018-09-11T19:03:29 [I] TenantManagerSingletonActor/TENANT_GUARDIAN_TERMINATED
```
* The first column is a timestamp, parseable by `dateutil.parser.parse`. Everything after the timestamp is treated as a text string.
* `event_plotter.py` automatically chooses the axis width (in hours) and the histogram bucket width (in minutes) based on the range of timestamps in the file. In the example above, the timestamps span about 19 hours on 2018-Sep-11.
* The output looks best when the timestamps span anywhere from an hour or two to about four weeks: buckets are never chosen smaller than one minute, and axes never display more than a week's worth of data.
* The program counts the number of the same message throughout the file, and chooses Matplotlib's `tab20` colors based on the frequency of the message. In the above output, the `[W] TenantManagerSingletonActor/START_REQUEST` message occurred the most (2288 total times) throughout the file (as shown in the legend at the bottom of the graph), followed by `[W] TenantManagerSingletonActor/NON_ROOT_TENANT_STOPPED` (1764 times) and so on.

## Useful command line options

`-a INT`, `--axis INT`

The number of hours that each axis should span. Must be one of 1, 2, 3, 4, 6, 12, 24, 48, or 168. If unspecified, an intelligent default is chosen, based on the timestamp range in the file.

`-b INT`, `--bucket INT`

The number of minutes per histogram bucket. Must be one of 1, 2, 5, 10, 15, 2, 30, or 60. If unspecified, an intelligent default is chosen.

`-n INT`, `--top-n INT`

How many of the top messages to display. Must be a value between 1 and 18. Default is 18. Messages which fall outside the top N are bucketized into a single bucket, `(all other messages)`.

`-q`, `--equal-y-axes`

If specified, then make every y-axis have the same vertical scale. Otherwise, let each y-axis scale itself according to the maximum histogram height on the axis.

`-y STR`, `--ylabel STR`

The text label for each y-axis.

`-t STR`, `--title STR`

The title for each axis.

`-s STR`, `--suptitle STR`

The overall graph title (i.e., the supertitle).

`-o FILENAME`, `--output-file FILENAME`

The filename to which to write the output. The type of file (PNG, GIF, etc.) is determined by Matplotlib based on the file extension (`.png`, `.gif`, etc.).
