import numpy as np
import gym
import environment.utils as utils
from gym import spaces
from environment.Elevator import Elevator
from environment.Passenger import Passenger



class Building(gym.Env):
    '''
    Building class to represent a building with multiple floors and elevators.
    Each elevator can move up or down, load passengers from the current floor, and unload passengers at the destination floor.
    The building can generate random passengers on each floor and keep track of the number of passengers in the building.
    The environment can be reset to an initial state, and the current state can be observed.
    Elevator classes are wrapped in the Building class and the states and actions are defined
    '''
    def __init__(self, elevator_count: int, max_floor: int, floor_capacity: int, elevator_capacity: int = 10, render_mode:str = "human", step_gen_flag: bool = False):
        '''
        elevator_count : int : Number of elevators in the building
        floor_capacity : int : Maximum number of passengers on each floor
        elevator_capacity : int : Maximum number of passengers in each elevator
        max_floor : int : Number of floors in the building
        render_mode : str : Mode for rendering the environment (default is "human")
        '''
        super(Building, self).__init__()
        self.elevator_count = elevator_count
        self.floor_capacity = floor_capacity
        self.elevator_capacity = elevator_capacity
        self.elevators = [Elevator(idx, elevator_capacity, max_floor) for idx in range(elevator_count)]
        self.max_floor = max_floor
        self.floor_info = [[] for _ in range(max_floor)]
        self.total_passengers = 0
        self.arrived_passengers = [[] for _ in range(max_floor)]

        # Passengers from file
        self.flag = False
        self.passengers = []

        # Step-Generation flag
        self.step_generation = step_gen_flag
        
        # Define observation space as a dictionary
        self.observation_space = spaces.Dict({
            "floor_passengers": spaces.Box(low=-1, high=floor_capacity, shape=(max_floor, max_floor), dtype=np.int32),
            "elevator_floors": spaces.Box(low=0, high=max_floor-1, shape=(elevator_count,), dtype=np.int32),
            "elevator_passengers": spaces.Box(low=0, high=elevator_capacity, shape=(elevator_count, max_floor), dtype=np.int32),
        })
        
        # Define action space: 0 (down), 1 (up), 2 (load), 3 (unload) for each elevator
        self.action_space = spaces.MultiDiscrete([4] * elevator_count)
        
        # Initialize for easier tracking
        self.cumulated_reward = 0
        self.render_mode = render_mode

        # Episode step tracking
        self.max_steps = utils.MAXSTEPS
        self.current_step = 0
        

        # keep of expected arrived passengers on each floor
        self.expected_arrived_passengers = [[] for _ in range(self.max_floor)]
        # extra info: list of passengers unloaded per elevator, loaded and waiting time in elevators for each passenger
        self.info = {"unloaded" : {}, "loaded" : {}, "waiting_time_in_elv" : {}}

    def set_flag(self, flag: bool = False, passengers: list = None, step_generation: bool = False):
        self.flag = flag
        self.passengers = passengers
        self.step_generation = step_generation

    def _get_observation(self):
        '''Return the current state in the observation_space format'''
        # Passengers in each floor
        floor_passengers = [[[floor, p.get_dest_floor()] for p in passengers] for floor, passengers in enumerate(self.floor_info)]
        floor_passengers = [x for x in floor_passengers if x != []]
        floor_passengers = [y for x in floor_passengers for y in x]
        floor_passengers = sorted(floor_passengers, key = lambda x: (x[0], x[1]))
        if len(floor_passengers) == 0:
            floor_passengers.append([-1, -1])
        floor_mat = np.zeros((self.max_floor, self.max_floor), dtype=np.int32)
        for origin, dest in floor_passengers:
            floor_mat[origin, dest] += 1
        floor_mat = np.clip(floor_mat, -1, self.floor_capacity)

        # Floor elevators at
        elev_floor = np.array([e.cur_floor for e in self.elevators], dtype=np.int32)

        # Passengers in each Elevator
        elevator_passengers = [e.get_dest_info() for e in self.elevators]
        elev_pass = np.zeros((self.elevator_count, self.max_floor), dtype=np.int32)
        for idx, passengers in enumerate(elevator_passengers):
            for passenger in passengers:
                elev_pass[idx, passenger] += 1
        elev_pass = np.clip(elev_pass, 0, self.elevator_capacity)

        return {
            "floor_passengers": floor_mat,
            "elevator_floors": elev_floor,
            "elevator_passengers": elev_pass
        }

    def get_all_remaining_passengers(self):
        # return sum(self.get_remain_passengers_in_elv(x) for x in self.elevators) + self.get_remain_passengers_in_building()
        remaining_passengers = 0

        #Add passengers left in elevators
        for e in self.elevators:
            remaining_passengers += e.passenger_count

        #Add passengers left to be picked from floors
        for floor in self.floor_info:
            remaining_passengers += len(floor)
        return remaining_passengers

    def generate_passengers(self, prob: float, passenger_max_num: int = 6):
        '''Generate random passengers in the building'''
        if(self.flag):
            # generate the passengers in the passengers list
            for p in self.passengers:
                cur_floor, dest_floor = p
                if cur_floor < self.max_floor and dest_floor < self.max_floor:
                    self.total_passengers += 1
                    new_passenger = Passenger(cur_floor=cur_floor, max_floor=self.max_floor, cur_step=self.current_step, passenger_idx=self.total_passengers, dest_floor=dest_floor)
                    self.floor_info[cur_floor].append(new_passenger)
                    self.expected_arrived_passengers[dest_floor].append(new_passenger)
        else:
            for floor_num in range(self.max_floor):
                if np.random.random() < prob and len(self.floor_info[floor_num]) < self.floor_capacity:
                    passenger_num = np.random.randint(1, passenger_max_num)
                    passenger_num = min(self.floor_capacity - len(self.floor_info[floor_num]), passenger_num)
                    additional_passengers = []
                    for i in range(passenger_num):
                        self.total_passengers += 1
                        additional_passengers.append(Passenger(cur_floor=floor_num, max_floor=self.max_floor, cur_step=self.current_step, passenger_idx=self.total_passengers))
                        self.expected_arrived_passengers[additional_passengers[-1].dest_floor].append(additional_passengers[-1])
                    self.floor_info[floor_num].extend(additional_passengers)

    def perform_action(self, action: list):
        # arrived_passengers_num_lst = []
        penalty = 0
        for idx, e in enumerate(self.elevators):
            if action[idx] == 0:
                if(e.move_down() == False):
                    penalty += utils.PENALTY_OUTOFBOUNDS
            elif action[idx] == 1:
                if(e.move_up() == False):
                    penalty += utils.PENALTY_OUTOFBOUNDS
            elif action[idx] == 2:
                if len(self.floor_info[e.cur_floor]) == 0:
                    penalty += utils.PENALTY_USELESS
                self.floor_info[e.cur_floor], loaded_passengers = e.load(self.floor_info[e.cur_floor])
                self.info["loaded"][e.idx] = [p.get_passenger_idx() for p in loaded_passengers]
            elif action[idx] == 3:
                unloaded_passengers = e.unload()
                if len(unloaded_passengers) == 0:
                    penalty += utils.PENALTY_USELESS
                # arrived_passengers_num_lst.append(len(unloaded_passengers))
                self.info["unloaded"][e.idx] = [p.get_passenger_idx() for p in unloaded_passengers]
                # for each elevator make a pair of idx_passenger and waiting time
                self.info["waiting_time_in_elv"][e.idx] = []
                for p in unloaded_passengers:
                    self.info["waiting_time_in_elv"][e.idx].append((p.get_passenger_idx(), p.get_wait_time(self.current_step)))
                
                self.arrived_passengers[e.cur_floor].extend(unloaded_passengers)
        
        penalty += self.get_all_remaining_passengers()
        reward = - penalty
        return reward

    def render_shafts(self):
        """
        Draw a vertical shaft diagram in the terminal. For each floor (top→0),
        prints:
        [Floor ##] │ Shaft0       Shaft1   … ShaftN-1 │ Waiting: […] │ Arrived: […]
        - E{i}[…] marks elevator i & its onboard passenger IDs
        - Waiting: IDs queued on this floor
        - Arrived: IDs that got off here this timestep
        """
        n_shafts = len(self.elevators)
        shaft_w = int(2.5 * self.elevator_capacity) + 5
        floor_label_w = 12
        waiting_w = int(self.floor_capacity * 2.5) + 12
        arrived_w = int(self.floor_capacity * 2.5)+ 12
        expected_arrived_w = self.floor_capacity * 2 + 12

        total_w = floor_label_w + 1 + n_shafts * shaft_w + 1 + waiting_w + 3 + arrived_w + 3 + expected_arrived_w
        sep = "=" * total_w

        for floor in reversed(range(self.max_floor)):
            # Build floor label
            lbl = f" Floor {floor:02d} "
            line = lbl.ljust(floor_label_w) + "│"

            # Elevator shafts
            for e in self.elevators:
                if e.cur_floor == floor:
                    p_ids = ",".join(str(p.get_passenger_idx()) for p in e.passengers_in_elv) or "-"
                    content = f"E{e.idx}[{p_ids}]"
                else:
                    content = ""
                line += content.center(shaft_w)
            line += "│"

            # Waiting
            w_ids = ",".join(str(p.get_passenger_idx()) for p in self.floor_info[floor])
            wait_str = f" Waiting: [{w_ids}] " if w_ids else " Waiting: [] "
            line += wait_str.ljust(waiting_w)

            # Arrived (just unloaded this tick)
            # You need to track unload events per floor; assume self.arrived[floor] is a list of IDs
            # a_ids = ",".join(str(pid) for pid in self.arrived.get(floor, []))
            a_ids = ",".join(str(p.get_passenger_idx()) for p in self.arrived_passengers[floor])
            arr_str = f" Arrived: [{a_ids}] " if a_ids else " Arrived: [] "
            line += arr_str.ljust(arrived_w)

            # Expected Arrived (just unloaded this tick)
            exa_ids = ",".join(str(p.get_passenger_idx()) for p in self.expected_arrived_passengers[floor])
            exa_str = f" Expected: [{exa_ids}] " if exa_ids else " Expected: [] "
            line += exa_str.ljust(expected_arrived_w)
            
            print(sep)
            print(line)
        print(sep)
        print("Step: ", self.current_step)
        print("Total # of people: ", self.total_passengers)
        print("People left to move: ", self.get_all_remaining_passengers())
        print("Passenger Loaded: ", self.info["loaded"])
        print("Passenger Unloaded: ", self.info["unloaded"])

    def reset(self):
        '''Reset the environment to an initial state'''
        for e in self.elevators:
            e.empty()
        self.floor_info = [[] for _ in range(self.max_floor)]
        self.info = {"unloaded" : {}, "loaded" : {}, "waiting_time_in_elv" : {}}
        self.current_step = 0
        self.total_passengers = 0
        self.cumulated_reward = 0
        self.arrived_passengers = [[] for _ in range(self.max_floor)]
        self.expected_arrived_passengers = [[] for _ in range(self.max_floor)]
        self.generate_passengers(prob=utils.SPAWN_PROB, passenger_max_num=min(self.floor_capacity, utils.GENERATION_RATE))  # Initial passengers
        return self._get_observation()

    def step(self, action):
        '''Take an action, update the environment, and return (obs, reward, done, info)'''
        self.current_step += 1
        if(self.step_generation and (self.current_step % utils.GENERATION_INTERVAL) == 0 and self.current_step <= utils.GENERATION_LIMIT):
            self.generate_passengers(prob=utils.SPAWN_INTERMEDIATE_PROB, passenger_max_num=min(self.floor_capacity, utils.SPAWN_INTERMEDIATE))
  
        self.info['unloaded'] = {}
        self.info['loaded'] = {}
        done = self.current_step >= self.max_steps or self.get_all_remaining_passengers() == 0
        reward = self.perform_action(action)
        obs = self._get_observation()  
        reward += done * utils.FINAL_REWARD  # Extra reward for finishing the episode
        info = {
            "remaining_passengers": self.get_all_remaining_passengers(),
            "reward": reward
        }
        return obs, reward, done, info


    def render(self, mode='human'):
        '''Render the environment (optional)'''
        if self.render_mode == 'human':
            self.render_shafts()
        else:
            raise ValueError("Invalid render mode. Use 'human' or 'ansi'.")