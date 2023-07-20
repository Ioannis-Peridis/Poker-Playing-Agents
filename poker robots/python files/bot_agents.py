import numpy as np

class BotAgent:
    def __init__(self, name):
        """Conctructor of BotAgent"""
        self.name = name


    def random_action(self, legal_actions):
        """Function that returns a random action from the legal actions"""
        return np.random.choice(legal_actions)

    def threshold_loose_action(self,round, strength, legal_actions):
        """Function for threshold loose agent"""
        if (round == 1):
            if(strength > 2):
                if('raise' in legal_actions):
                    return 'raise'
                elif('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            else:
                if('check' in legal_actions):
                    return 'check'
                else:
                    return 'call'
        elif (round == 2):
            if(strength >= 4):
                if('raise' in legal_actions):
                    return 'raise'
                elif('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            elif(strength < 4):
                if (np.random.random()<0.6):
                    if('raise' in legal_actions):
                        return 'raise'
                    elif('bet' in legal_actions):
                        return 'bet'
                    else:
                        return 'call'
                else:
                    if('check' in legal_actions):
                        return 'check'
                    else:
                        return 'fold'
            
    def threshold_tight_action(self,round, strength, legal_actions):
        """Function for tight threshold tight agent"""
        if (round == 1):
            if(strength > 3):
                if('raise' in legal_actions):
                    return 'raise'
                elif('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            elif(strength == 3):
                if('check' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            else:
                if('check' in legal_actions):
                    return 'check'
                else:
                    return 'fold'
        elif (round == 2):
            if(strength > 5):
                if('raise' in legal_actions):
                    return 'raise'
                elif('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            else:
                if('check' in legal_actions):
                    return 'check'
                else:
                    return 'fold'

    def super_human_action(self, round, strength, legal_actions):
        if (round == 1):
            if(strength >= 4):
                if('raise' in legal_actions):
                    return 'raise'
                elif('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            elif(strength == 3):
                if (np.random.random()<0.5):
                    if('raise' in legal_actions):
                        return 'raise'
                    elif('bet' in legal_actions):
                        return 'bet'
                    else:
                        return 'call'
                else:
                    if('check' in legal_actions):
                        return 'check'
                    else:
                        return 'call'
            elif(strength == (1 or 2)):
                    if('check' in legal_actions):
                        return 'check'
                    else:
                        return 'fold'   
        elif (round == 2):
            if(strength > 5):
                if('raise' in legal_actions):
                    return 'raise'
                elif('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            elif( strength == ( 4 or 5)):
                if('bet' in legal_actions):
                    return 'bet'
                else:
                    return 'call'
            elif( strength < 4):
                if('check' in legal_actions):
                        return 'check'
                else:
                        return 'fold'
            else:
                return np.random.choice(legal_actions)