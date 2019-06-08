#create a feature "TimeCat" -{0-3}
import time
from time import mktime
from datetime import datetime

def getTimeCat(Datetime):
    # extract time categories
    str_time = datetime.strptime(Datetime, "%m/%j/%y %H:%M").time()
    ts = datetime.fromtimestamp(mktime(str_time))

    # --> Morning = 0400-1000
    mornStart = datetime.time(4, 0, 1)
    mornEnd = datetime.time(10, 0, 0)

    # --> Midday = 1000-1600
    midStart = datetime.time(10, 0, 1)
    midEnd = datetime.time(16, 0, 0)

    # --> Evening = 1600-2200
    eveStart = datetime.time(16, 0, 1)
    eveEnd = datetime.time(22, 0, 0)

    # --> Late Night = 2200-0400
    lateStart = datetime.time(22, 0, 1)
    lateEnd = datetime.time(4, 0, 0)

    if time_in_range(mornStart, mornEnd, ts):
      timecat = 0 #morning
    elif time_in_range(midStart, midEnd, ts):
      timecat = 1 #midday
    elif time_in_range(eveStart, eveEnd, ts):
      timecat = 2 #evening
    elif time_in_range(lateStart, lateEnd, ts):
      timecat = 3 #late night

    return timecat