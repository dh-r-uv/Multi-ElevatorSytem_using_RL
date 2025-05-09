import random

class Passenger(object):
    '''
    Passenger class to represent a passenger in the elevator system.
    Each passenger has a current floor, a destination floor, and a spawn step.
    The passenger can be generated randomly in the building / can also be generated using input parameters.
    '''
    def __init__(self, cur_floor : int, max_floor : int, cur_step : int, passenger_idx :int, dest_floor : int = None):
        self.cur_floor = cur_floor
        self.dest_floor = dest_floor if dest_floor is not None else random.choice(list(range(cur_floor)) + list(range(cur_floor+1,max_floor)))
        self.spawn_step = cur_step
        self.passenger_idx = passenger_idx
        
    def get_dest_floor(self) -> int:
        return self.dest_floor
    
    def get_cur_floor(self) -> int:
        return self.cur_floor
    
    def get_wait_time(self, cur_step : int) -> int:
        return cur_step - self.spawn_step

    def get_passenger_idx(self) -> int:
        return self.passenger_idx
