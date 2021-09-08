from datetime import datetime, time
from typing import List


def hours_to_time(hours: List[int]) -> List[time]:
    '''
        :param hours: a list of hours as a time of day (e.g. [1, 16] is 01:00 and 16:00 o'clock)
        :return: a list of datetime time objects that correspond to the hours in hours
    '''
    # TODO-lev: These must be utc time
    return [datetime.strptime(str(t), '%H').time() for t in hours]
