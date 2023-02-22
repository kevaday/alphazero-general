import pyautogui as pg
import time
import keyboard
p = (1500, 524)

# time.sleep(1)
# pg.moveTo(p)
# time.sleep(0.5)
# pg.click()
# time.sleep(0.5)

# keyboard.press_and_release('ctrl+break')
# time.sleep(3)
# keyboard.write("python -m alphazero.envs.othello.train")
# keyboard.press_and_release("enter")
#time.sleep(45 * 60)
first = True
while True:
	time.sleep(1)
	pg.moveTo(p)
	time.sleep(0.5)
	pg.click()
	time.sleep(0.5)

	keyboard.press_and_release('ctrl+break')
	if not first:
		time.sleep(25 * 60)
	time.sleep(5)
	keyboard.write("python -m alphazero.envs.reversi_othello.train")
	keyboard.press_and_release("enter")
	for i in range(int(4.5 * 60 * 60)):
		print(i)
		time.sleep(1)
	first = False
