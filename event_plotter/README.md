# Plot a histogram of events

Given a text file with a timestamp at the start of each line and a message on the rest of the line, plot a histogram of the most commonly-occurring messages.

Sample data is found in `test/events_all.txt` and `test/events_detail.txt`. For reasonable data (time ranges between a few minutes and a couple of weeks, messages with a reasonable amount of repetition in them), no arguments are necessary:
```
python event_plotter.py test/events_detail.txt
```
Output is to `events.png` in current working directory.
