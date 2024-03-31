# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:27:53 2018

@author: mona
"""

# CS161 - Hawana
# SUM 2018
# HW 4
# Jon Baird

from random import choice as cs

def total(hand):
    
    
# how many aces in the hand
    aces = hand.count(11)
# the ace can be 11 or 1
# this little while loop figures it out for you
    s = sum(hand)
    if s > 21 and aces > 0: # you have gone over 21 but there is an ace
      while aces > 0 and s > 21:
          
           s -= 10
           aces -= 1
            
    return (s)

if __name__ == "__main__":
    
# a suit of cards in blackjack assume the following values
    cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

# there are 4 suits per deck and usually several decks
# so we assume the cards list to be an unlimited pool

    dwin = 0 # dealer win counter
    pwin = 0 # player win counter
    i=0;
    while i<10:
        
        player = []
        player.append(cs(cards))# draw 2 cards for the player to start
        player.append(cs(cards))
        pbust = False # player busted flag
        dbust = False # dealer busted flag
   # while True:
                  # loop for the player's play ...
    
        pt = total(player)
        print ("The player has these cards %s with a total value of %d" % (player, pt))
        if pt > 21:
            
            print ("--> The player is busted!")
            pbust = True
            #break
        elif pt == 21:
            
           print ("\a BLACKJACK!!! YOU WIN!!!")
            #break
        else:
            
           Hit = str(input("Hit or Stand/Done (h or s): ")).lower()            
            
           if 'h' in Hit:
            
              player.append(cs(cards))
        
          
       # while True: 
            # loop for the dealer's play ...
        dealer = []#dealer generally stands around 17 or 18
    #while True:
            
        dealer.append(cs(cards))
        dealer.append(cs(cards))
        dt = total(dealer)
        print ("The dealer has these cards %s with a total value of %d" % (dealer, dt))
            
        if dt < 18:
                
           dealer.append(cs(cards))
        else:
                #code to determine winner
            if dt > 21:
                    
                print ("--> The dealer is busted!")
                dbust = True
                    #break
            if pbust == False:
                    
                   print ("The player wins!")
                   pwin += 1
            elif dt > pt:
                    
                print ("The dealer wins!")
                dwin += 1
            elif dt == pt:
                print ("It's a draw!")
            elif pt > dt:
                if pbust == False:
                    print ("The player wins!")
                    pwin += 1
                elif dbust == False:
                    print ("The dealer wins!")
                    dwin += 1
                       # break
  
        print ("Wins, player = %d dealer = %d" % (pwin, dwin))
        exit = str(input("Press Enter (q to quit): ")).lower()
        if 'q' in exit:
              # break;
             print ("\n\nThanks for playing blackjack!");
             break;
