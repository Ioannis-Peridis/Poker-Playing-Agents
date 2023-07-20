import numpy as np
from bot_agents import BotAgent

class QLearningAgent:

    def __init__(self,learning_rate=0.3, discount=0.9, epsilon=0.1):
        """Initialize the QLearning Agent"""

        # Initialize all the values for learning rate, gamma, epsilon
        self.name = "QLearning Agent"
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

        # Generate the state space
        self.state_space = self.generate_state_space()

        # available actions
        self.actions = ['fold', 'check', 'bet', 'call', 'raise']

        # initialize q table as zeros
        self.q_table = np.zeros((len(self.state_space), len(self.actions)))
        # boolean value to see if the agent is in training or not is needed for some prints
        self.in_training = True



    def generate_state_space(self):
        """Function that initializes the state space."""
        # Preflop 0-4 states
        # Flop  5-9 with High card
        #       10-14 with pair
        #       15-19 with set  
        state_space = {}
        for i in range(20):
            state_space[i] = i
        return state_space
    
    def pick_action(self, state, legal_actions, epsilon):
        """Function to pick action It uses ε-greedy algorithm."""
        # get q value for current state
        action_val = self.q_table[self.state_space[state]]
        # get q values only for legal actions
        legal_action_val = [action_val[action] for action in range(len(legal_actions))]
        
        # explore using a random action
        if(np.random.uniform(0,1) < epsilon):
            action = legal_actions[np.random.choice(len(legal_action_val),1)[0]]
        else:
            action = legal_actions[np.argmax(legal_action_val)]
        return action


    def convert_action_into_number(self, action):
        """Function to convert action to number"""
        # fold = 0
        # check = 1
        # call = 2
        # bet = 3
        # raise = 4

        if(action == 'fold'):
            return 0
        elif(action == 'check'):
            return 1
        elif(action == 'call'):
            return 2
        elif(action == 'bet'):
            return 3
        elif(action == 'raise'):
            return 4
        else:
            return 0

    def update_q_table(self, state, action, reward, next_state):
        # 
        action_num = self.convert_action_into_number(action)
        st = self.state_space[state]
        current_q = self.q_table[st ,action_num]
        max_q = np.max(self.q_table[self.state_space[next_state]])
        new_q = current_q + self.learning_rate * (self.discount * max_q - current_q + reward)
        self.q_table[self.state_space[state], action_num] = new_q
    
    def print_q_table(self):
        print("State                | Action               | Q-Value")
        print("-----------------------------------------------------")
        for state in self.state_space:
            s = self.state_space[state]
            for a, action in enumerate(self.actions):
                tmp = str(state)
                print("{:<20} | {:<20} | {:.2f}".format(tmp, action, self.q_table[s][a]), flush=True)
    

    def train_agent(self,game, opponent_type, num_episodes = 20000):
        """Function that trains our agent against an opponent. You will run num_episodes rounds"""

        # put starting stack very high so that there is no need for reseting every round
        starting_stack = 100000
        # the bot you're playing against
        bot_agent = BotAgent(opponent_type)
        # create 2 players
        player_one, player_two = game.create_players(starting_stack,'giannis','tzortzis')
        # pointer to see which player plays 1st in each round
        goes_first = 0
        # loop for training
        for episode in range(num_episodes):
            # play a round
            self.run_episode(game,starting_stack,player_one,player_two,bot_agent,goes_first)
            # change who is going to play first in the next round
            goes_first += 1
            # self.epsilon = (episode+1)**(-1/4)

    

    def state(self, strength, round):
        # Function that tells in which state we are depending in the hand and the round we are on
        if(round == 1):
            return strength - 1
        else:
            return strength + 4

    def run_episode(self,game,starting_stack,player_one,player_two, bot_agent,goes_first):
        """Function that simulates a round in a poker game"""

        # initialize some actions
        starting_actions = ['bet', 'check']
        actions_after_check = ['check', 'bet']
        actions_after_raise = ['call', 'fold']
        actions_after_bet = ['call', 'fold', 'raise']
        round = 1
        # create the deck
        deck = game.create_deck()
        while(True):
            # if in round 1(pre flop then see who is playing first or second)
            if(round == 1):
                pot = 1

                if(goes_first % 2 == 0):
                    plays_first = player_one
                    plays_first.opponent = False
                    plays_second = player_two
                    plays_second.opponent = True
                else:
                    plays_first = player_two
                    plays_first.opponent = True
                    plays_second = player_one
                    plays_second.opponent = False

                # deal cards in the players and the board
                [plays_first.hand, plays_second.hand, board] = game.deal_cards(deck)
                # restore the deck for next usage
                deck = game.restore_deck([plays_first.hand, plays_second.hand, board[0], board[1]], deck)

            # ask the player who speaks 1st his action. If he is our agebt it goes to pick_action
            plays_first.action = game.ask_action_for_training(plays_first.opponent, starting_actions, bot_agent, round, game.calculate_hand_strength(plays_first.hand, board,round),self)
            # if plays_first is our player update q table otherwise do nothing
            if(plays_first.opponent == False):
                next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                self.update_q_table(next_state, plays_first.action, 0, next_state)
            if(plays_first.action == 'check'):
                plays_second.action = game.ask_action_for_training(plays_second.opponent, actions_after_check, bot_agent, round, game.calculate_hand_strength(plays_second.hand, board ,round),self)
                if(plays_second.action == 'check'):
                    # if check check sequence and in round 1 then
                    if (round == 1):
                        # get current state
                        current_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                        # change round
                        round = 2
                        # if plays_second is our player get the next state and update the q table
                        if(plays_second.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                            self.update_q_table(current_state, plays_second.action, 0, next_state)
                    # if check check sequence and in round 2 then we are at showdown
                    elif(round == 2):
                        # calcualte winner
                        winner = game.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                        plays_first.stack, plays_second.stack , reward = game.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                        # if plays_second is our player get the next state and update the q table
                        if(plays_second.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                            # if split then reward = 0
                            if(winner == 'split'):
                                self.update_q_table(next_state,plays_second.action,0,next_state)
                            # if winner = plays_first then -reward
                            elif(winner == 'player_one'):
                                self.update_q_table(next_state,plays_second.action,-reward,next_state)
                            # else winner = p2 then +reward
                            else:
                                self.update_q_table(next_state, plays_second.action, reward, next_state)

                        # if you are p1 then get the rewards depending if you won or not
                        elif(plays_first.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                            if( winner == 'split'):
                                self.update_q_table(next_state,plays_first.action,0,next_state)
                            elif(winner == 'player_one'):
                                self.update_q_table(next_state,plays_first.action,reward,next_state)
                            else:
                                self.update_q_table(next_state, plays_first.action, -reward, next_state)
                        round = 1 
                        return  
                elif(plays_second.action == 'bet'):
                    # increment 1 to pot if action is bet
                    pot += 1
                    # if my agent = p2 udate q table for the action bet
                    if(plays_second.opponent == False):
                        next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                        self.update_q_table(next_state, plays_second.action, 0, next_state)
                    plays_first.action = game.ask_action_for_training(plays_first.opponent, actions_after_raise, bot_agent, round, game.calculate_hand_strength(plays_first.hand, board,round),self)
                    if(plays_first.action == 'call'):
                        # increment pot
                        pot += 1
                        # if in round 1 and sequence bet call
                        if (round == 1):
                            # get curr state 
                            current_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                            round = 2 
                            # if p1 = our agent then get next state and update q table
                            if(plays_first.opponent == False):
                                next_state = self.state(game.calculate_hand_strength(plays_first.hand, board, round), round)
                                self.update_q_table(current_state, plays_first.action, 0, next_state)
                        #  if in round 2 then calculate winner and update q table of your player according to the winnings
                        elif(round == 2):
                            winner = game.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                            plays_first.stack, plays_second.stack, reward = game.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                            if(plays_second.opponent == False):
                                next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                                if(winner == 'split'):
                                    self.update_q_table(next_state,plays_second.action,0,next_state)
                                elif(winner == 'player_one'):
                                    self.update_q_table(next_state,plays_second.action, -reward,next_state)
                                else:
                                    self.update_q_table(next_state, plays_second.action, reward, next_state)
                            if(plays_first.opponent == False):
                                next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                                if( winner == 'split'):
                                    self.update_q_table(next_state,plays_first.action,0,next_state)
                                elif(winner == 'player_one'):
                                    self.update_q_table(next_state,plays_first.action,reward,next_state)
                                else:
                                    self.update_q_table(next_state, plays_first.action, -reward, next_state)
                            round = 1
                            return
                    # if sequence bet-fold then update q table of your player. If you folded the update with -q reward else +reward
                    elif(plays_first.action == 'fold'):
                        plays_first.stack, plays_second.stack, reward = game.calculate_blinds(pot, 'player_two', True, plays_first.stack, plays_second.stack, starting_stack)
                        if(plays_first.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_first.hand, board, round), round)
                            self.update_q_table(next_state,plays_first.action, -reward, next_state)
                        elif(plays_second.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                            self.update_q_table(next_state, plays_second.action, reward, next_state)
                        round = 1
                        return
            
            elif(plays_first.action == 'bet'):
                # increment pot by 1 
                pot += 1
                # get p2 action
                plays_second.action = game.ask_action_for_training(plays_second.opponent, actions_after_bet, bot_agent, round, game.calculate_hand_strength(plays_second.hand, board,round),self)
                # if p2 action = call 
                if(plays_second.action == 'call'):
                    pot += 1
                    # if in round 1 and sequence bet call then 
                    if (round == 1):
                        current_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                        # go to next round
                        round = 2 
                        # if my agent is p2 then update q table for action call in round 1
                        if(plays_second.opponent == False):
                            # get next state
                            next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                            # update q table
                            self.update_q_table(current_state, plays_second.action, 0, next_state)
                    # if in round 2 and sequence bet-call
                    elif(round == 2):
                        # go to showdown
                        winner = game.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                        plays_first.stack, plays_second.stack, reward = game.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                        # depending on the winner and which player you are update q table with the proportional +/-reward
                        if(plays_second.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                            if(winner == 'split'):
                                self.update_q_table(next_state, plays_second.action, 0, next_state)
                            elif(winner == 'player_one'):
                                self.update_q_table(next_state, plays_second.action, -reward, next_state)
                            else:
                                self.update_q_table(next_state, plays_second.action, reward, next_state)

                        elif(plays_first.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                            if(winner == 'split'):
                                self.update_q_table(next_state, plays_first.action, 0, next_state)
                            elif(winner == 'player_one'):
                                self.update_q_table(next_state, plays_first.action, reward, next_state)
                            else:
                                self.update_q_table(next_state, plays_first.action, -reward, next_state)                
                        
                        round = 1
                        return
                
                # if avction bet and p2 folds then update q table accordingly
                # if you folded update q table with -reward else with +reward
                elif(plays_second.action == 'fold'):
                    plays_first.stack, plays_second.stack, reward = game.calculate_blinds(pot, 'player_one', True, plays_first.stack, plays_second.stack, starting_stack)
                    if(plays_second.opponent == False):
                        next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                        self.update_q_table(next_state, plays_second.action, -reward, next_state)
                    elif(plays_first.opponent == False):
                        next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                        self.update_q_table(next_state, plays_first.action, reward, next_state)     
                    
                    round = 1
                    return
                
                # if bet-raise sequence then
                elif(plays_second.action == 'raise'):
                    # increment pot by 
                    pot += 2
                    # if you are the one who raised you juct took an action so update q table
                    if(plays_second.opponent == False):
                        next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                        self.update_q_table(next_state, plays_second.action, 0, next_state)     
                    
                    plays_first.action = game.ask_action_for_training(plays_first.opponent, actions_after_raise, bot_agent, round, game.calculate_hand_strength(plays_first.hand, board,round),self)
                    # if bet-raise-call
                    if(plays_first.action == 'call'):
                        pot += 1
                        # if in round 1 Change round and if you are the one who called update q table by taking the next state 
                        if (round == 1):
                            current_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                            round = 2 
                            if(plays_first.opponent == False):
                                next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                                self.update_q_table(current_state, plays_first.action, 0, next_state) 
                        # if in round 2 then you are at showdown
                        elif(round == 2):
                            # calculate winner
                            winner = game.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                            plays_first.stack, plays_second.stack, reward = game.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                            # according to who you are then update q table
                            if(plays_first.opponent == False):
                                next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                                if(winner == 'split'):
                                    self.update_q_table(next_state, plays_first.action, 0, next_state) 
                                elif(winner == 'player_one'):
                                    self.update_q_table(next_state, plays_first.action, reward, next_state) 
                                else:
                                    self.update_q_table(next_state, plays_first.action, -reward, next_state) 
                            elif(plays_second.opponent == False):
                                next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                                if(winner == 'split'):
                                    self.update_q_table(next_state, plays_second.action, 0, next_state) 
                                elif(winner == 'player_one'):
                                    self.update_q_table(next_state, plays_second.action, -reward, next_state) 
                                else:
                                    self.update_q_table(next_state, plays_second.action, reward, next_state) 

                            round = 1
                            return 
                    # if bet-raise-fold then update q table and and the episode 
                    elif(plays_first.action == 'fold'):
                        plays_first.stack, plays_second.stack, reward = game.calculate_blinds(pot, 'player_two', True, plays_first.stack, plays_second.stack, starting_stack)
                        if(plays_first.opponent == False):
                            next_state = self.state(game.calculate_hand_strength(plays_first.hand, board,round), round)
                            self.update_q_table(next_state, plays_first.action, -reward, next_state) 
                        else:
                            next_state = self.state(game.calculate_hand_strength(plays_second.hand, board,round), round)
                            self.update_q_table(next_state, plays_second.action, reward, next_state) 

                        round = 1 
                        return
