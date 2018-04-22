import os
import time
import tensorflow as tf
from PIL import ImageGrab

debug =  1

def screen_grab(save = False, debug = False):
	window_ul = (0, 0) #upperleft corner of window, width then depth
	window_width = 928
	window_depth = 666
	window = (window_ul[0], window_ul[1], window_width, window_depth)
	window_img = ImageGrab.grab(window)
	#deck
	deck_ul_ratio = (0.3, 0.3)
	deck_lr_ratio = (0.7, 0.5)
	deck_ul = (window_width * deck_ul_ratio[0], window_depth * deck_ul_ratio[1])
	deck_lr = (window_width * deck_lr_ratio[0], window_depth * deck_lr_ratio[1])
	deck = deck_ul + deck_lr #concat tuple
	deck_img = window_img.crop(deck)
	read_from_deck(deck_img, save = save, debug = debug)
	#my_hands
	my_ul_ratio = (0.40, 0.57) #captures bet
	my_lr_ratio = (0.60, 0.80)
	my_ul = (window_width * my_ul_ratio[0], window_depth * my_ul_ratio[1])
	my_lr = (window_width * my_lr_ratio[0], window_depth * my_lr_ratio[1])
	my_cor = my_ul + my_lr
	my_img = window_img.crop(my_cor)
	#my action options
	action_ul_ratio = (0.47, 0.76)  #capture hand strength
	action_lr_ratio = (1.0, 1.0)
	action_ul = (window_width * action_ul_ratio[0], window_depth * action_ul_ratio[1])
	action_lr = (window_width * action_lr_ratio[0], window_depth * action_lr_ratio[1])
	action = action_ul + action_lr #concat tuple
	action_img = window_img.crop(action)
	available_actions = detect_actions(action_img)
	#save
	if (save):
		#deck_img.show()
		#my_img.show()
		deck_img.save(os.getcwd() + r'\deck_' + str(int(time.time())) + '.png', 'PNG')
		#my_img.save(os.getcwd() + r'\my_' + str(int(time.time())) + '.png', 'PNG');
		#action_img.show()
	    #window_img.save(os.getcwd() + '\full_snap__' + str(int(time.time())) + '.png', 'PNG')
	return len(available_actions)

def detect_actions(action_img, debug = False):
	#change to foreach
	first_action = detect_action(action_img, 0.25)
	second_action = detect_action(action_img, 0.5)
	third_action = detect_action(action_img, 0.75)
	ret = set()
	if (first_action):
		ret.add(first_action)
	if (second_action):
		ret.add(second_action)
	if (third_action):
		ret.add(third_action)
	return ret
	
def detect_action(action_img, ordinate, debug = False):
	action_img = action_img.load()
	#print(action_img[30,30])
	return True
	
def collect_train():
	while True:
		screen_grab(save=True)
		time.sleep(30)

def read_from_deck(deck_img, save = False, debug=True):
	start_up_left = [26, 35]
	card_width = 57
	card_height = 80
	card_width_gap = 65
	for i in range(5):
		lr = [start_up_left[0] + card_width, start_up_left[1] + card_height]
		card_img = deck_img.crop(start_up_left + lr)
		if save:
			card_img.save(os.getcwd() + r'\card_' + str(int(time.time())) + str(i) + '.png', 'PNG')
		start_up_left[0] += card_width_gap
		card_result = reco_card(card_img)
		if debug:
			card_img.show()
			print(card_result)
		
def reco_card(card_img):
	pass
		
def main():
	screen_grab(debug=True)
	#while True:
		#my_turn = screen_grab()
		#sleep(0.05)
		#if my_turn:
			#pass
			#cal21()
 
if __name__ == '__main__':
	main()

