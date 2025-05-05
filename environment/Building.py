# Building.py
import gym
from gym import spaces
import numpy as np
from Elevator import Elevator
from Passenger import Passenger

class Building(gym.Env):
    '''
    Gym environment wrapping the original Building logic.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 total_elevator_num: int,
                 max_floor: int,
                 max_passengers_in_floor: int,
                 max_passengers_in_elevator: int,
                 elevator_capacity: int = 10):
        super(Building, self).__init__()
        self.remain_passengers_num = 0
        self.cumulated_reward = 0
        self.total_elevator_num = total_elevator_num
        self.max_passengers_in_floor = max_passengers_in_floor
        self.max_passengers_in_elevator = max_passengers_in_elevator
        self.capacity = elevator_capacity
        self.elevators = [Elevator(idx, elevator_capacity, max_floor)
                          for idx in range(total_elevator_num)]
        self.max_floor = max_floor
        self.floors_information = [[] for _ in range(max_floor)]
        self.spawn_prob = 0.1
        self.step_count = 0

        # Gym spaces
        self.action_space = spaces.MultiDiscrete([4] * self.total_elevator_num)
        self.observation_space = spaces.Dict({
            'waiting': spaces.Box(
                low=0, high=self.max_passengers_in_floor,
                shape=(self.max_floor, self.max_floor-1), dtype=np.int32),
            'elevator_pass': spaces.Box(
                low=-1, high=self.max_floor-1,
                shape=(self.total_elevator_num, self.capacity), dtype=np.int32),
            'elevator_floor': spaces.Box(
                low=0, high=self.max_floor-1,
                shape=(self.total_elevator_num,), dtype=np.int32)
        })

        # generate random passengers
        self.generate_passengers(self.spawn_prob)

    def seed(self, seed=None):
        """Set random seed for environment."""
        np.random.seed(seed)
        return [seed]

    def reset(self):
        self.empty_building()
        self.step_count = 0
        self.generate_passengers(self.spawn_prob)
        return self._build_obs()

    def step(self, action):
        reward = self.perform_action(list(action))
        self.generate_passengers(self.spawn_prob)
        obs = self._build_obs()
        self.step_count += 1
        done = (self.get_remain_all_passengers() == 0)
        return obs, reward, done, {'step': self.step_count}

    def render(self, mode='human'):
        self.print_building(self.step_count)

    def _build_obs(self):
        wait = np.zeros((self.max_floor, self.max_floor-1), dtype=np.int32)
        for floor_idx, plist in enumerate(self.floors_information):
            for p in plist:
                dest = p.get_dest()
                j = dest if dest < floor_idx else dest - 1
                wait[floor_idx, j] += 1
        # ep = -np.ones((self.total_elevator_num, self.capacity), dtype=np.int32), dtype=np.int32)
        ep = np.ones((self.total_elevator_num, self.capacity), dtype=np.int32) * -1
        for i, e in enumerate(self.elevators):
            for j, p in enumerate(e.curr_passengers_in_elv):
                if j < self.capacity:
                    ep[i, j] = p.get_dest()
        ef = np.array([e.curr_floor for e in self.elevators], dtype=np.int32)
        return {'waiting': wait, 'elevator_pass': ep, 'elevator_floor': ef}

    # -- Original Building methods --
    def get_arrived_passengers(self) -> int:
        arrived = 0
        for e in self.elevators:
            arrived += e.arrived_passengers_num
            e.arrived_passengers_num = 0
        return arrived

    def get_state(self) -> tuple:
        floor_passengers = [
            [[floor, p.get_dest()] for p in plist]
            for floor, plist in enumerate(self.floors_information)
        ]
        floor_passengers = [y for x in floor_passengers if x for y in x]
        if not floor_passengers:
            floor_passengers = [[-1, -1]]
        elv_passengers = [e.get_passengers_info() for e in self.elevators]
        elv_passengers = [x for x in elv_passengers if x]
        if not elv_passengers:
            elv_passengers = [[-1]]
        elevators_floors = [e.curr_floor for e in self.elevators]
        return floor_passengers, elv_passengers, elevators_floors

    def empty_building(self):
        self.floors_information = [[] for _ in range(self.max_floor)]
        for e in self.elevators:
            e.empty()
        self.remain_passengers_num = 0

    def get_remain_passengers_in_building(self) -> int:
        return sum(len(floor) for floor in self.floors_information)

    def get_remain_passengers_in_elv(self, elv) -> int:
        return len(elv.curr_passengers_in_elv)

    def get_remain_all_passengers(self) -> int:
        return (sum(self.get_remain_passengers_in_elv(e) for e in self.elevators)
                + self.get_remain_passengers_in_building())

    def generate_passengers(self, prob: float, passenger_max_num: int = 6):
        for floor_num in range(self.max_floor):
            if (np.random.random() < prob
                    and len(self.floors_information[floor_num]) < self.max_passengers_in_floor):
                num = np.random.randint(1, passenger_max_num)
                num = min(self.max_passengers_in_floor,
                          len(self.floors_information[floor_num]) + num)
                new = [Passenger(now_floor=floor_num, max_floor=self.max_floor)
                       for _ in range(num)]
                self.floors_information[floor_num] += new
                self.remain_passengers_num += num

    def perform_action(self, action: list) -> float:
        arrived_list = []
        penalty_list = []
        for idx, e in enumerate(self.elevators):
            a = action[idx]
            if a == 0:
                if e.curr_floor == 0:
                    penalty_list.append(-1)
                e.move_down()
            elif a == 1:
                if e.curr_floor == e.max_floor-1:
                    penalty_list.append(-1)
                e.move_up()
            elif a == 2:
                if not self.floors_information[e.curr_floor]:
                    penalty_list.append(-1)
                self.floors_information[e.curr_floor] = (
                    e.load_passengers(self.floors_information[e.curr_floor]))
            elif a == 3:
                arrived = e.unload_passengers(self.floors_information[e.curr_floor])
                if arrived == 0:
                    penalty_list.append(-1)
                arrived_list.append(arrived)
        reward = sum(penalty_list) - self.get_remain_all_passengers()
        return reward

    def print_building(self, step : int):
        for idx in reversed(list(range(1,self.max_floor))):
            print("=======================================================")
            print("= Floor #%02d ="%idx, end=' ')
            for e in self.elevators:
                if e.curr_floor == idx:
                    print("  Lift #%d"%e.idx, end=' ')
                else:
                    print("         ", end=' ')
            print(" ")
            print("=  Waiting  =", end=' ')
            for e in self.elevators:
                if e.curr_floor == idx:
                    print("    %02d   "%len(e.curr_passengers_in_elv), end=' ')
                else:
                    print("          ", end=' ')
            print(" ")
            print("=    %03d    ="%len(self.floors_information[idx]))
        print("=======================================================")
        print("= Floor #00 =", end=' ')
        for e in self.elevators:
            if e.curr_floor == 0:
                print("  Lift #%d"%e.idx, end=' ')
            else:
                print("         ", end=' ')
        print(" ")
        print("=  Arrived  =", end=' ')
        for e in self.elevators:
            if e.curr_floor == 0:
                print("    %02d   "%len(e.curr_passengers_in_elv), end=' ')
            else:
                print("          ", end=' ')		
        print(" ")
        print("=    %03d    ="%len(self.floors_information[0]))
        print("=======================================================")
        print("")
        print("People to move: %d "%(self.remain_passengers_num - len(self.floors_information[0])))
        print("Total # of people: %d"%self.remain_passengers_num)
        print("Step: %d"%step)
        print('state : ',self.get_state())
        #print('now reward : ',self.get_reward())

# agent.py

