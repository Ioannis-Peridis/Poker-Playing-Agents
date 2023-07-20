from player import Player
from bot_agents import BotAgent
from policy_iteration import PolicyIterationAgent
from qlearning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt 

class Game:
    def __init__(self):
        """Constructor of poker game"""
        self.name = "poker"

    def create_deck(self):
        """
        Function that creates the deck for every possible card and every possible suit

        Returns:
            (list): The state of the player
        """
        deck = []
        # The suits
        suit_list = ['S', 'H', 'D', 'C']
        # The cards
        rank_list = ['T', 'J', 'Q', 'K', 'A']
        # FOr every possible scenario append to deck
        for suit in suit_list:
            for rank in rank_list:
                card = f'{rank} of {suit}'
                deck.append(card)
        return deck

    def deal_cards(self, deck):
        """
        Function that deals the cards for the round
        
        Args:
            deck (list): A list of the deck

        Returns:
            (handp1, handp2, board): returns the 
        """

        cards = []
        # Deal 4 and remove them from the deck
        for i in range(4):
            card = np.random.choice(deck)
            cards.append(card)
            deck.remove(card)
        return cards[0], cards[1], cards[2:4] 

    def print_card(self, hand, player):
        """
        Function used to print the cards beautifully
        
        Args:
            hand (list): The hand suit and number
            player (str): Player name

        """

        suit = self.find_suit(hand)
        print(f"{player}'s Hand:")
        print("----------------")
        print("┌─────────┐")
        print(f"│{hand[0]}{suit}       │")
        print("│         │")
        print("│         │")
        print(f"│    {suit}    │")
        print("│         │")
        print("│         │")
        print(f"│       {hand[0]}{suit}│")
        print("└─────────┘")

    def print_board(self, board):
        """
        Function that prints the board beautifully taken by rlcard

        Args:
            board (list): The board list contains 2 cards

        """

        suit_one = self.find_suit(board[0])
        suit_two = self.find_suit(board[1])
        print("This is the Board:")
        print("---------------------------")
        print("┌─────────┐", end="  ")
        print("┌─────────┐")
        print(f"│{board[0][0]}{suit_one}       │", end="  ")
        print(f"│{board[1][0]}{suit_two}       │")
        print("│         │", end="  ")
        print("│         │")
        print("│         │", end="  ")
        print("│         │")
        print(f"│    {suit_one}    │", end="  ")
        print(f"│    {suit_two}    │")
        print("│         │", end="  ")
        print("│         │")
        print("│         │", end="  ")
        print("│         │")
        print(f"│       {board[0][0]}{suit_one}│", end="  ")
        print(f"│       {board[1][0]}{suit_two}│")
        print("└─────────┘", end="  ")
        print("└─────────┘")

    def find_suit(self, hand):
        """
        Function that converts the character of the suit
        
        Args:
            hand (list): the hand suit and number

        Returns:
            (str): the symbol of the suit
        """

        if (hand[5] == 'C'):
            suit = '♣'
        elif(hand[5] == 'H'):
            suit = '♥'
        elif(hand[5] == 'S'):
            suit = '♠'
        elif(hand[5] == 'D'):
            suit = '♦'
        return suit

    def create_players(self, starting_stack, name_one, name_two):
        """
        Function that creates the 2 players

        Args:
            starting_stack (int): The staring stack to initialize the stack of the players
            name_one (str): Name of player 1 
            name_two (str): Name of player 2

        Returns:
            (dict): The state of the player
        """
        # player1
        player_one = Player(name_one, starting_stack)
        # player2
        player_two = Player(name_two, starting_stack)

        return player_one, player_two

    def game_not_over(self, stack_one, stack_two, name_one, name_two):
        """
        Function that checks that the game is over or not

        Args:
            stack_one (int): Stack of Player 1
            stack_two (str): Stack of Player 2
            name_one (str): Name of player 1
            name_two (str): Name of player 2

        Returns:
            (boolean): true if game is over. False otherwise
        """    
        if(stack_one > 0 and stack_two > 0):
            return True
        else:
            print("=================================")
            print("********** Game is over *********")
            print("=================================")
            if(stack_one <= 0):
                print(f"\n{name_one} Looses | {name_two} Wins!\n")
            elif(stack_two <= 0):
                print(f"\n{name_two} Looses | {name_one} Wins!\n")
            return False
        

    def restore_deck(self, cards, deck):
        """
        Function that restores the deck. Meaning that returns to the deck the hand's of the players and the board
        
        Args:
            cards (list): Cards to restore
            deck  (list): the list to extend

        Returns:
            (boolean): true if game is over. False otherwise
        """         
        deck.extend(cards)
        return deck

    def calculate_blinds(self, total_pot, winner, not_showdown, stack_one, stack_two, start_stack):
        """
        Function that calculates the new stacks after a round

        Args:
            total_pot (int): Total pot
            winner (str): The winner of the round
            not_showdown (boolean): Boolean value that shows if we went to showdon or not. If we did not go to showdown that means that someone folded
            stack_one (int): stack of p1
            stack_two (int): stack of p2
            start_stack (int): starting stack

        Returns:
            (boolean): true if game is over. False otherwise
        """ 

        # if someone folded take the last betting blind 
        if (not_showdown == True):
            total_pot -= 1
        # if p1 won calculate blinds
        if (winner == 'player_one'):
            stack_one = stack_one + total_pot/2
            stack_two = stack_two - total_pot/2
        # if p2 won calculate blinds
        elif (winner == "player_two"):
            stack_one = stack_one - total_pot/2
            stack_two = stack_two + total_pot/2
        # if split then stacks are the as last round
        elif (winner == 'split'):
            stack_one = stack_one
            stack_two = stack_two
        # last 2 cases are for game over
        if(stack_one < 0):
            stack_one = 0
            stack_two = start_stack * 2
        elif(stack_two < 0):
            stack_two = 0
            stack_one = start_stack * 2
        return stack_one, stack_two, total_pot/2

    def calculate_hand_strength(self, hand, board, round):
        """
        Function that calculates the strength of the hand

        Args:
            hand (list): hand of the player
            board (list): The board
            round (int): which round in the game we are (Preflop or flop)

        Returns:
            (int): The stregth of the hand
        """ 
        # if round 1 then return 1-5 for the hand you have
        if (hand[0] == 'T'): strength = 1
        elif(hand[0] =='J'): strength = 2
        elif(hand[0] =='Q'): strength = 3
        elif(hand[0] =='K'): strength = 4
        elif(hand[0] =='A'): strength = 5
        if(round == 1):
            return strength
        
        # if you have a set strength is stregth of round 1 + 10
        if(hand[0] == board[0][0] and hand[0]  == board[1][0]):
            strength += 10
        # if you have a pair strength is stregth of round 1 + 5
        elif((hand[0] == board[0][0] and hand != board[1][0]) or (hand[0]  == board[1][0] and hand[0]  != board[0][0])):
            strength += 5
        return strength

    def calculate_winner(self, hand_one, hand_two, board,round):
        """
        Function that calculates the winner of the round

        Args:
            hand_one (list): hand of p2
            hand_two (list): hand of p2
            board (list): board
            round (int): the round
        Returns:
            (str): the winner player_one if p1 wins, player_two if p2 wins, split if it is a tie
        """ 

        # calculate the strength of the 2 hands
        p1 = self.calculate_hand_strength(hand_one, board,round)
        p2 = self.calculate_hand_strength(hand_two, board,round)

        # compare the 2 strengths
        if(p1 > p2):
            winner = 'player_one'
        elif(p1 < p2):
            winner = 'player_two'
        elif(p1 == p2):
            winner = 'split'
        return winner


    def ask_action(self, opponent,legal_actions, bot_agent, round, strength, my_agent):
        """
        Function like step. Chooses the action according to the agent. If bot choose function from there. Otherwise choose from Qlearning Agent or Policy Iteration

        Args:
            opponent (boolean): If you are opponent then variabke is true. False otherwise
            legal_actions (list): list of legal actions
            bot_agent (object): object of BotAgent
            round (int): the round
            strength (int): strength of the hand
            my_agent (Object): Policy Iteration agent or Qlearning agent
        Returns:
            (str): The action chosen
        """         
        if (opponent == True and bot_agent.name == 'random'):
            action = bot_agent.random_action(legal_actions)
            print(f"Opponent's action is : {action}")
        elif(opponent == True and bot_agent.name == 'threshold_loose'):
            action = bot_agent.threshold_loose_action(round, strength, legal_actions)
            print(f"Opponent's action is : {action}")
        elif(opponent == True and bot_agent.name == 'threshold_tight'):
            action = bot_agent.threshold_tight_action(round, strength, legal_actions)
            print(f"Opponent's action is : {action}")
        elif(opponent == True and bot_agent.name == 'superhuman'):
            action = bot_agent.super_human_action(round, strength, legal_actions)
            print(f"Opponent's action is : {action}")
        elif(my_agent.name == 'Policy Iteration Agent'):
            v,pi = my_agent.policy_iteration(my_agent.P)
            pi = {s: pi(s) for s in range(len(my_agent.P))}  #keep the old policy to compare with new
            action  = self.choose_action(strength, pi, round, legal_actions, bot_agent.name)
            print(f"My action is : {action}")
        elif(my_agent.name == 'QLearning Agent'): 
            action = my_agent.pick_action(strength, legal_actions, my_agent.epsilon)
            print(f"My action is : {action}")

        return action
    
    def ask_action_for_training(self, opponent,legal_actions, bot_agent, round, strength, my_agent):
        """
        Function like ask_action but the only difference is that the prints are substracted

        Args:
            opponent (boolean): If you are opponent then variabke is true. False otherwise
            legal_actions (list): list of legal actions
            bot_agent (object): object of BotAgent
            round (int): the round
            strength (int): strength of the hand
            my_agent (Object): Policy Iteration agent or Qlearning agent
        Returns:
            (str): The action chosen
        """    

        if (opponent == True and bot_agent.name == 'random'):
            action = bot_agent.random_action(legal_actions)
        elif(opponent == True and bot_agent.name == 'threshold_loose'):
            action = bot_agent.threshold_loose_action(round, strength, legal_actions)
        elif(opponent == True and bot_agent.name == 'threshold_tight'):
            action = bot_agent.threshold_tight_action(round, strength, legal_actions)
        elif(opponent == True and bot_agent.name == 'superhuman'):
            action = bot_agent.super_human_action(round, strength, legal_actions)
        elif(my_agent.name == 'Policy Iteration Agent'):
            v,pi = my_agent.policy_iteration(my_agent.P)
            pi = {s: pi(s) for s in range(len(my_agent.P))}  #keep the old policy to compare with new
            action  = self.choose_action(strength, pi, round, legal_actions, bot_agent.name)
        elif(my_agent.name == 'QLearning Agent'): 
            action = my_agent.pick_action(strength, legal_actions, my_agent.epsilon)

        # if(not my_agent.in_training):
        #     print(f"Action is : {action}")
        return action

    def convert_action(self, action_choice):
        """
        Function that converts an integer to the proportional action

        Args:
            action_choice (int): integer that means which action to take and is converted to a string
    
        Returns:
            (str): The action as a string
        """  
                    
        if(action_choice == 0):action = 'fold'
        elif(action_choice == 1):action= 'check'
        elif(action_choice ==  2): action = 'call'
        elif(action_choice ==  3): action ='bet'
        elif(action_choice ==  4): action ='raise'
        return action

    def choose_action(self, strength, pi, round, legal_actions, opponent_type):
        """
        Like ask_action but for the bots

        Args:
            legal_actions (int): 
            strength (int): 
            pi (int): 
            round (int): 
            opponent_type (int):  
    
        Returns:
            (str): The action as a string
        """  
        if( round == 1):
            action = self.convert_action(pi[strength - 1])
        elif(round == 2):
            action = self.convert_action(pi[strength + 4])
        if( action in legal_actions):
            return action
        else:
            if( action == 'fold'):
                return 'check'
            elif( action == 'check'):
                return 'fold'
            elif(action == 'bet'):
                if('raise' in legal_actions):
                    return 'raise'
                else:
                    return 'call'
            elif( action == 'raise'):
                if('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            elif( action == 'call'):
                if(opponent_type == 'threshold_tight' or opponent_type == 'random'):
                    if('bet' in legal_actions):
                        return 'bet'
                    else:
                        return 'check'
                elif(opponent_type == 'threshold_loose'):
                    if('bet' in legal_actions):
                        return 'check'
                    else:
                        return 'bet'          

    def hand_end(self, pot,one,two,stack_one,stack_two,showdown,folded):
        """
        Function to print the stacks at the end of every round

        Args:
            pot (int): total pot
            one (str): name of player 1
            two (str): name of player 2
            stack_one (int): stack of p1
            stack_two (int):  stack of p2
            showdown (boolean):  True if game reached to showdown. False otherwise
            folded (boolean):  True if someone folded. False otherwise
    
        """  
        if(showdown == True):
            print(f"\nHand ending going to Showdown :")
        else:
            print(f"\nHand ending {folded} Folded :")
        print("-------------------------------")
        print(f"Total Pot : {pot}")
        print(f"{one}'s Stack : {stack_one}")
        print(f"{two}'s Stack : {stack_two}\n")
        return

    def update_reward(self, first,winner,total_reward,reward):
        """
        Function to update the rewards used for the experiments(plots)

        Args:
            first (boolean): true if i play first 
            winner (str): Who won 
            total_reward (int): Total reward
            reward (int): Reward of the round 
    
        Returns:
            (int): The new total reward
        """     
        # I play first  
        if(first == False):
            if(winner == 'player_one'):
                total_reward = total_reward + reward
            elif(winner == 'player_two'):
                total_reward = total_reward - reward
        # I play second
        else:
            if(winner == 'player_one'):
                total_reward = total_reward - reward
            elif(winner == 'player_two'):
                total_reward = total_reward + reward
        return total_reward

    def main(self):
        """
        Main function of the game

        """  
        starting_stack = 20
        opponent_type = 'threshold_loose'
        bot_agent = BotAgent(opponent_type)
        # agent = PolicyIterationAgent(opponent_type)
        agent = QLearningAgent()
        player_one, player_two = self.create_players(starting_stack,'giannis','tzortzis')
        deck = self.create_deck()
        starting_actions = ['bet', 'check']
        actions_after_check = ['check', 'bet']
        actions_after_raise = ['call', 'fold']
        actions_after_bet = ['call', 'fold', 'raise']

        goes_first = 0
        round = 1
        iteration = 0
        pot = 1
        reward = []
        average = []
        total_reward = 0
        blinds_per_hand = 0
        # train qlearning agent
        agent.train_agent(self,opponent_type)
        # print q table to see the results
        agent.print_q_table()
        input()

        # flag in_training false
        agent.in_training = False
        agent.epsilon = 0


        print("\n===========================================================")
        print("******************** SIMPLE POKER GAME ********************")
        print("===========================================================\n")
        # loop of the game
        while( self.game_not_over(player_one.stack, player_two.stack, player_one.name, player_two.name)):
            # If in round 1 find who plays 1st or 2nd and change some flags needed
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
                # change game pointer so that in the next game the other player plays 1st
                goes_first += 1

                print(".........................................................\n("+plays_first.name+" : plays first this turn)")
                print("==================")
                print(f"Going to Hand {iteration} :")
                iteration += 1
                print("==================\n")
                print("********** ROUND 1 **********")
                
                # Deal cards and print them in the console
                [plays_first.hand, plays_second.hand, board] = self.deal_cards(deck)
                self.print_card(plays_first.hand,plays_first.name)
                self.print_card(plays_second.hand,plays_second.name)
                deck = self.restore_deck([plays_first.hand, plays_second.hand, board[0], board[1]], deck)

            # if in round 2 print the board
            if(round == 2):
                print("\n********** ROUND 2 **********")
                self.print_board(board)

            # ask action from player 1
            plays_first.action = self.ask_action(plays_first.opponent, starting_actions, bot_agent, round, self.calculate_hand_strength(plays_first.hand, board,round),agent)
            if(plays_first.action == 'check'):
                # ask action from p2
                plays_second.action = self.ask_action(plays_second.opponent, actions_after_check, bot_agent, round, self.calculate_hand_strength(plays_second.hand, board,round),agent)
                if(plays_second.action == 'check'):
                    # if check-check sequence and in round 1 then go to next round
                    if (round == 1):
                        round = 2 
                    # if in round 2 and check-check then go to showdown
                    elif(round == 2):
                        # see who won
                        winner = self.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                        # calculate the new stacks of each player
                        plays_first.stack, plays_second.stack,y = self.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                        
                        #rewards to be plotted later for the results 
                        reward.append(y)
                        total_reward = self.update_reward(plays_first.opponent,winner,total_reward,reward[iteration-1])
                        average.append(total_reward/iteration)
                        
                        #print hand end 
                        self.hand_end(pot,plays_first.name,plays_second.name,plays_first.stack,plays_second.stack,True,plays_first.name)
                        round = 1   
                elif(plays_second.action == 'bet'):
                    pot += 1
                    #plays_first.action =  input(f"{plays_first.name} chooses action : {actions_after_raise}\n")
                    plays_first.action = self.ask_action(plays_first.opponent, actions_after_raise, bot_agent, round, self.calculate_hand_strength(plays_first.hand, board,round),agent)
                    
                    # part of the tree check-bet-call
                    if(plays_first.action == 'call'):
                        pot += 1
                        if (round == 1):
                            round = 2 
                        elif(round == 2):
                            winner = self.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                            plays_first.stack, plays_second.stack, y= self.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                            reward.append(y)
                            total_reward = self.update_reward(plays_first.opponent,winner,total_reward,reward[iteration-1])
                            average.append(total_reward/iteration)
                            self.hand_end(pot,plays_first.name,plays_second.name,plays_first.stack,plays_second.stack,True,plays_first.name)
                            round = 1
                    # check-bet-fold
                    elif(plays_first.action == 'fold'):
                        plays_first.stack, plays_second.stack,y = self.calculate_blinds(pot, 'player_two', True, plays_first.stack, plays_second.stack, starting_stack)
                        reward.append(y)
                        total_reward = self.update_reward(plays_first.opponent,'player_two',total_reward,reward[iteration-1])
                        average.append(total_reward/iteration)
                        self.hand_end(pot,plays_first.name,plays_second.name,plays_first.stack,plays_second.stack,False,plays_first.name)
                        round = 1

            # if action of the first player = bet. New branch of the tree
            elif(plays_first.action == 'bet'):
                pot += 1
                #plays_second.action = input(f"{plays_second.name} chooses action : {actions_after_bet}\n")
                plays_second.action = self.ask_action(plays_second.opponent, actions_after_bet, bot_agent, round, self.calculate_hand_strength(plays_second.hand, board,round),agent)
                # bet-call sequence = showdown in round 2 else change round
                if(plays_second.action == 'call'):
                    pot += 1
                    if (round == 1):
                        round = 2 
                    elif(round == 2):
                        winner = self.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                        plays_first.stack, plays_second.stack,y= self.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                        reward.append(y)
                        total_reward = self.update_reward(plays_first.opponent,winner,total_reward,reward[iteration-1])
                        average.append(total_reward/iteration)
                        self.hand_end(pot,plays_first.name,plays_second.name,plays_first.stack,plays_second.stack,True,plays_first.name)
                        round = 1
                # bet-fold is end of hand 
                elif(plays_second.action == 'fold'):
                    plays_first.stack, plays_second.stack,y = self.calculate_blinds(pot, 'player_one', True, plays_first.stack, plays_second.stack, starting_stack)
                    reward.append(y)
                    total_reward = self.update_reward(plays_first.opponent,'player_one',total_reward,reward[iteration-1])
                    average.append(total_reward/iteration)
                    self.hand_end(pot,plays_first.name,plays_second.name,plays_first.stack,plays_second.stack,False,plays_second.name)
                    round = 1
                # bet-raise
                elif(plays_second.action == 'raise'):
                    pot += 2
                    #plays_first.action =  input(f"{plays_first.name} chooses action : {actions_after_raise}\n")
                    plays_first.action = self.ask_action(plays_first.opponent, actions_after_raise, bot_agent, round, self.calculate_hand_strength(plays_first.hand, board,round),agent)
                    # bet-raise-call. If in round 1 go to round 2. if in round 2 go to showdown
                    if(plays_first.action == 'call'):
                        pot += 1
                        if (round == 1):
                                round = 2 
                        elif(round == 2):
                                winner = self.calculate_winner(plays_first.hand, plays_second.hand, board,round)
                                plays_first.stack, plays_second.stack, y = self.calculate_blinds(pot, winner, False, plays_first.stack, plays_second.stack, starting_stack)
                                reward.append(y)
                                total_reward = self.update_reward(plays_first.opponent,winner,total_reward,reward[iteration-1])
                                average.append(total_reward/iteration)
                                self.hand_end(pot,plays_first.name,plays_second.name,plays_first.stack,plays_second.stack,True,plays_first.name)
                                round = 1 
                    # bet-raise-fold. End of hand
                    elif(plays_first.action == 'fold'):
                        plays_first.stack, plays_second.stack, y = self.calculate_blinds(pot, 'player_two', True, plays_first.stack, plays_second.stack, starting_stack)
                        reward.append(y)
                        total_reward = self.update_reward(plays_first.opponent,winner,total_reward,reward[iteration-1])
                        average.append(total_reward/iteration )
                        self.hand_end(pot,plays_first.name,plays_second.name,plays_first.stack,plays_second.stack,False,plays_first.name)
                        round = 1

        blinds_per_hand = starting_stack / iteration
        print(f"Average Blinds/Hand Win Ratio : {blinds_per_hand}")
        plt.title("Average Reward") 
        plt.xlabel("Number of Hands") 
        plt.ylabel("Total Cumulative Reward") 
        plt.plot(range(len(average)),average)
        plt.show()
                              
if __name__ == "__main__":
    poker_game = Game()
    poker_game.main()