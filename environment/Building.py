import numpy as np
import gym
from gym import spaces
from environment.Elevator import Elevator
from environment.Passenger import Passenger

class Building(gym.Env):
    '''
    Building controls elevators and passengers.
    It sets constraints and operates entire environment, adapted for Gym.
    '''
    def __init__(self, total_elevator_num: int, max_floor: int, max_passengers_in_floor: int, max_passengers_in_elevator: int, elevator_capacity: int = 10, render_mode:str = "human"):
        '''
        remain_passengers_num(int): Remaining passengers in building
        total_elevator_num(int): Total number of elevators
        max_passengers_in_floor(int): Maximum number of passengers on one floor
        max_passengers_in_elevator(int): Maximum number of passengers in one elevator
        max_floor(int): Maximum floor in the building
        floors_information(list(Passenger)): Passenger information on each floor
        elevators(list(Elevator)): List of elevators
        '''
        super(Building, self).__init__()
        self.remain_passengers_num = 0
        self.cumulated_reward = 0
        self.total_elevator_num = total_elevator_num
        self.max_passengers_in_floor = max_passengers_in_floor
        self.max_passengers_in_elevator = max_passengers_in_elevator
        self.elevators = [Elevator(idx, elevator_capacity, max_floor) for idx in range(total_elevator_num)]
        self.max_floor = max_floor
        self.floors_information = [[] for _ in range(max_floor)]
        self.render_mode = render_mode
        
        # Define observation space as a dictionary
        self.observation_space = spaces.Dict({
            "floor_passengers": spaces.Box(low=0, high=max_passengers_in_floor, shape=(max_floor, max_floor), dtype=np.int32),
            "elevator_floors": spaces.Box(low=0, high=max_floor-1, shape=(total_elevator_num,), dtype=np.int32),
            "elevator_passengers": spaces.Box(low=0, high=max_passengers_in_elevator, shape=(total_elevator_num, max_floor), dtype=np.int32),
        })
        
        # Define action space: 0 (down), 1 (up), 2 (load), 3 (unload) for each elevator
        self.action_space = spaces.MultiDiscrete([4] * total_elevator_num)
        
        # Episode step tracking
        self.max_steps = 1000
        self.current_step = 0

    def get_arrived_passengers(self) -> int:
        arrived_passengers = 0
        for e in self.elevators:
            arrived_passengers += e.arrived_passengers_num
            e.arrived_passengers_num = 0
        return arrived_passengers

    def get_state(self) -> tuple:
        floor_passengers = [[[floor, passenger.get_dest()] for passenger in passengers] for floor, passengers in enumerate(self.floors_information)]
        floor_passengers = [x for x in floor_passengers if x != []]
        floor_passengers = [y for x in floor_passengers for y in x]
        if len(floor_passengers) == 0:
            floor_passengers.append([-1, -1])
        elv_passengers = [e.get_passengers_info() for e in self.elevators]
        elv_passengers = [x for x in elv_passengers if x != []]
        if len(elv_passengers) == 0:
            elv_passengers.append([-1])
        elevators_floors = [e.curr_floor for e in self.elevators]
        return floor_passengers, elv_passengers, elevators_floors

    def empty_building(self):
        '''Clears the building'''
        self.floors_information = [[] for _ in range(self.max_floor)]
        for e in self.elevators:
            e.empty()
        self.remain_passengers_num = 0

    def get_remain_passengers_in_building(self):
        return sum(len(x) for x in self.floors_information)

    def get_remain_passengers_in_elv(self, elv):
        return len(elv.curr_passengers_in_elv)

    def get_remain_all_passengers(self):
        return sum(self.get_remain_passengers_in_elv(x) for x in self.elevators) + self.get_remain_passengers_in_building()

    def generate_passengers(self, prob: float, passenger_max_num: int = 6):
        '''Generate random passengers in the building'''
        for floor_num in range(self.max_floor):
            if np.random.random() < prob and len(self.floors_information[floor_num]) < self.max_passengers_in_floor:
                passenger_num = np.random.randint(1, passenger_max_num)
                passenger_num = min(self.max_passengers_in_floor - len(self.floors_information[floor_num]), passenger_num)
                additional_passengers = [Passenger(now_floor=floor_num, max_floor=self.max_floor) for _ in range(passenger_num)]
                self.floors_information[floor_num].extend(additional_passengers)
                self.remain_passengers_num += passenger_num

    def perform_action(self, action: list):
        arrived_passengers_num_lst = []
        penalty_lst = []
        for idx, e in enumerate(self.elevators):
            if action[idx] == 0:
                if e.curr_floor == 0:
                    penalty_lst.append(-1)
                e.move_down()
            elif action[idx] == 1:
                if e.curr_floor == (self.max_floor - 1):
                    penalty_lst.append(-1)
                e.move_up()
            elif action[idx] == 2:
                if len(self.floors_information[e.curr_floor]) == 0:
                    penalty_lst.append(-1)
                self.floors_information[e.curr_floor] = e.load_passengers(self.floors_information[e.curr_floor])
            elif action[idx] == 3:
                arrived_passengers_num = e.unload_passengers(self.floors_information[e.curr_floor])
                if arrived_passengers_num == 0:
                    penalty_lst.append(-1)
                arrived_passengers_num_lst.append(arrived_passengers_num)
        
        # Reward includes positive reward for arrivals
        reward = sum(arrived_passengers_num_lst) + sum(penalty_lst) - self.get_remain_all_passengers()
        return reward

    def print_building(self, step: int):
        for idx in reversed(range(1, self.max_floor)):
            print("=======================================================")
            print(f"= Floor #{idx:02d} =", end=' ')
            for e in self.elevators:
                print(f"  Lift #{e.idx}" if e.curr_floor == idx else "         ", end=' ')
            print(" ")
            print("=  Waiting  =", end=' ')
            for e in self.elevators:
                print(f"    {len(e.curr_passengers_in_elv):02d}   " if e.curr_floor == idx else "          ", end=' ')
            print(" ")
            print(f"=    {len(self.floors_information[idx]):03d}    =")
        print("=======================================================")
        print("= Floor #00 =", end=' ')
        for e in self.elevators:
            print(f"  Lift #{e.idx}" if e.curr_floor == 0 else "         ", end=' ')
        print(" ")
        print("=  Arrived  =", end=' ')
        for e in self.elevators:
            print(f"    {len(e.curr_passengers_in_elv):02d}   " if e.curr_floor == 0 else "          ", end=' ')
        print(" ")
        print(f"=    {len(self.floors_information[0]):03d}    =")
        print("=======================================================")
        print(f"\nPeople to move: {self.remain_passengers_num - len(self.floors_information[0])}")
        print(f"Total # of people: {self.remain_passengers_num}")
        print(f"Step: {step}")
        print('state : ', self.get_state())

    def reset(self):
        '''Reset the environment to an initial state'''
        self.empty_building()
        self.current_step = 0
        self.generate_passengers(prob=0.5)  # Initial passengers
        return self._get_observation()

    def step(self, action):
        '''Take an action, update the environment, and return (obs, reward, done, info)'''
        self.generate_passengers(prob=0.1)  # New passengers each step
        reward = self.perform_action(action)
        obs = self._get_observation()
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.get_remain_all_passengers() == 0
        info = {}
        return obs, reward, done, info

    def _get_observation(self):
        '''Return the current state in the observation_space format'''
        floor_passengers = np.zeros((self.max_floor, self.max_floor), dtype=np.int32)
        for floor in range(self.max_floor):
            for passenger in self.floors_information[floor]:
                dest = passenger.get_dest()
                floor_passengers[floor, dest] += 1

        elevator_floors = np.array([e.curr_floor for e in self.elevators], dtype=np.int32)
        
        elevator_passengers = np.zeros((self.total_elevator_num, self.max_floor), dtype=np.int32)
        for idx, e in enumerate(self.elevators):
            for p in e.curr_passengers_in_elv:
                dest = p.get_dest()
                elevator_passengers[idx, dest] += 1

        return {
            "floor_passengers": floor_passengers,
            "elevator_floors": elevator_floors,
            "elevator_passengers": elevator_passengers,
        }

    def render(self, mode='human'):
        '''Render the environment (optional)'''
        if self.render_mode == 'human':
            self.print_building(self.current_step)
        elif self.render_mode == 'ansi':
            return self.print_building(self.current_step)
        else:
            raise ValueError("Invalid render mode. Use 'human' or 'ansi'.")