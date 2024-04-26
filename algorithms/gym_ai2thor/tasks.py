"""
Different task implementations that can be defined inside an ai2thor environment
"""

from gym_ai2thor.utils import InvalidTaskParams


class BaseTask:
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.task_config = config
        self.max_episode_length = config.get('max_episode_length', 1000)
        # default reward is negative to encourage the agent to move more
        self.movement_reward = config.get('movement_reward', -0.01)
        self.step_num = 0

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """

        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class PickUpTask(BaseTask):
    """
    This task consists of picking up a target object. Rewards are only collected if the right
    object was added to the inventory with the action PickUp (See gym_ai2thor.envs.ai2thor_env for
    details). Because the agent can only carry 1 object at a time in its inventory, to receive
    a lot of reward one must learn to put objects down. Optimal behaviour will lead to the agent
    spamming PickupObject and PutObject near a receptacle. target_objects is a dict which contains
    the target objects which the agent gets reward for picking up and the amount of reward was the
    value
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # check that target objects are not selected as NON pickupables
        missing_objects = []
        for obj in kwargs['task']['target_objects'].keys():
            if obj not in kwargs['pickup_objects']:
                missing_objects.append(obj)
        if missing_objects:
            raise InvalidTaskParams('Error initializing PickUpTask. The objects {} are not '
                                    'pickupable!'.format(missing_objects))

        self.target_objects = kwargs['task'].get('target_objects', {'Mug': 1})
        self.prev_inventory = []

    def transition_reward(self, state):
        reward, done = self.movement_reward, False
        curr_inventory = state.metadata['inventoryObjects']
        object_picked_up = not self.prev_inventory and curr_inventory and \
                           curr_inventory[0]['objectType'] in self.target_objects

        if object_picked_up:
            # One of the Target objects has been picked up. Add reward from the specific object
            reward += self.target_objects.get(curr_inventory[0]['objectType'], 0)
            print('{} reward collected!'.format(reward))

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        self.prev_inventory = state.metadata['inventoryObjects']
        return reward, done

    def reset(self):
        self.prev_inventory = []
        self.step_num = 0

class PickUpPutTask(BaseTask):
    """
    This task consists of picking up a target object and putting it in a specified receptacle.
    Rewards are collected if the target object is picked up and successfully put in the receptacle.
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        # Check that target object is pickupable and acceptable receptacle is valid
        target_object = kwargs['task']['target_objects'].keys()
        if target_object not in kwargs['pickup_objects']:
            raise InvalidTaskParams('Error initializing PickUpPutTask. The target object {} '
                                    'is not pickupable!'.format(target_object))
        if kwargs['task']['put_receptacles'][0] not in kwargs['acceptable_receptacles']:
            raise InvalidTaskParams('Error initializing PickUpPutTask. The put receptacle {} '
                                    'is not acceptable!'.format(kwargs['task']['put_receptacles'][0]))

        self.target_object = target_object
        self.put_receptacle = kwargs['task']['put_receptacles'][0]
        self.prev_inventory = []

    def transition_reward(self, state):
        reward, done = self.movement_reward, False
        curr_inventory = state.metadata['inventoryObjects']
        object_picked_up = not self.prev_inventory and curr_inventory and \
                           curr_inventory[0]['objectType'] == self.target_object

        if object_picked_up and state.metadata['lastActionSuccess']:
            # The target object has been picked up successfully
            print('Picked up {}!'.format(self.target_object))

        if self.put_receptacle in state.metadata['objects'][0]['receptacleObjectIds'] and \
           state.metadata['lastActionSuccess']:
            # The target object has been successfully put in the receptacle
            print('Put {} in {}!'.format(self.target_object, self.put_receptacle))
            reward += 1  # Reward for successfully putting the object in the receptacle

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        self.prev_inventory = state.metadata['inventoryObjects']
        return reward, done

    def reset(self):
        self.prev_inventory = []
        self.step_num = 0

class PutObjectTask(BaseTask):
    """
    This task involves the agent putting a specific object into a designated receptacle. 
    The agent gets a reward for correctly placing the specified object into the correct receptacle.
    The task configuration should specify 'target_object' and 'destination_object' to guide the agent's actions.
    """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.target_object = kwargs['task'].get('target_object')
        self.destination_object = kwargs['task'].get('destination_object')

        if self.target_object not in kwargs['pickup_objects']:
            raise InvalidTaskParams('Error initializing PutObjectTask. The target object {} is not '
                                    'pickupable!'.format(self.target_object))
        if self.destination_object not in kwargs['acceptable_receptacles']:
            raise InvalidTaskParams('Error initializing PutObjectTask. The destination {} is not '
                                    'an acceptable receptacle!'.format(self.destination_object))

    def transition_reward(self, state):
        reward, done = self.movement_reward, False
        last_put = state.metadata.get('lastObjectPut')
        last_receptacle = state.metadata.get('lastObjectPutReceptacle')

        # Check if the last object put matches the target object and was put in the correct receptacle
        if last_put and last_receptacle:
            if last_put['objectType'] == self.target_object and last_receptacle['objectType'] == self.destination_object:
                reward += 1.0  # Assign a fixed reward for correct placement
                print('{} successfully placed in {}. Reward collected!'.format(self.target_object, self.destination_object))
                done = True  # Task completed successfully

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        return reward, done

    def reset(self):
        self.step_num = 0
        print('Task reset. Place the {} in the {}.'.format(self.target_object, self.destination_object))
