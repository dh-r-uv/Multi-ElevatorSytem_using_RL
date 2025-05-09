import numpy as np 

class Elevator(object):
    '''
    Elevator class to represent an elevator in the building.
    Each elevator has a current floor, a maximum number of passengers, and a list of passengers currently in the elevator.
    The elevator can move up or down, load passengers, and unload passengers.
    The elevator can also be emptied to reset its state. reset generates random passengers in the building.
    penalty is depended on waiting time of passengers in the elevator.
    Reward is given at the end of the destination of passengers.
    '''
    def __init__(self, idx : int, capacity : int, max_floor : int):
        '''
        idx: int : index of the elevator
        capacity: int : maximum number of passengers in the elevator
        max_floor: int : maximum number of floors in the building
        '''
        self.idx = idx
        self.max_floor = max_floor
        self.capacity = capacity
        self.cur_floor = 0
        self.passengers_in_elv = []
        self.passenger_count = 0

    def move_up(self):
        if self.cur_floor < self.max_floor-1:
            self.cur_floor += 1
            return True
        return False

    def move_down(self):
        if self.cur_floor > 0:
            self.cur_floor -= 1
            return True
        return False

    def empty(self):
        self.passengers_in_elv = []
        self.cur_floor = 0

    def load(self, passengers_in_floor : list) -> list:
        '''
        loads passengers into the elevator
        returns a list of passengers that could not be loaded into the elevator and a list of passengers that were loaded
        '''
        not_loaded_passengers = []
        can_load = min(self.capacity - self.passenger_count, len(passengers_in_floor))
        for p in passengers_in_floor[:can_load]:    # load these passengers
            self.passenger_count += 1
            self.passengers_in_elv.append(p)
        
        not_loaded_passengers = passengers_in_floor[can_load:]
        loaded_passengers = passengers_in_floor[:can_load]
        return not_loaded_passengers, loaded_passengers
    
    def get_dest_info(self) -> list :
        return [p.get_dest_floor() for p in self.passengers_in_elv]
    
    def unload(self) -> list:
        '''
        unloads passengers from the elevator
        returns a list of passengers that were unloaded
        '''
        unloaded_passengers_idx = []
        for i in range(self.passenger_count):
            if self.passengers_in_elv[i].dest_floor == self.cur_floor:
                unloaded_passengers_idx.append(i)
        
        unloaded_passengers_idx.reverse() 

        passengers_unloaded = []
        for i in unloaded_passengers_idx:
            passengers_unloaded.append(self.passengers_in_elv[i])
            self.passengers_in_elv.pop(i)
            self.passenger_count -= 1
        
        return passengers_unloaded