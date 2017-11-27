import os, sys, inspect

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)


cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"lib/Leap")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
    cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"lib/Anderson")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)


import Leap
import numpy as np
import pandas as pd
import math

import neuralnetworks as nn
import mlutils as ml

globletter = 0 #what letter was entered on keybaord

#*******************************************************
#https://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user
#makes getch platform independent.
class _Getch:
    #Gets a single character from standard input.  Does not echo to the screen.
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()
#****************************************************

#****************************************************
#leap motion sensor listener. on_frame acts as main interaction point.
class HandListener(Leap.Listener):

    def on_connect(self, controller):
        print("Connected")


    def on_frame(self, controller): #every time a frame is possible, this updates
        #print("Frame available")
        global globletter
        frame = controller.frame()  #get a frame
        rightHand = frame.hands.rightmost #get the right-most (right) hand in the sensor's field of view
        #print(rightHand)

        #TODO: add a timer here which prevents checking the hand until the next interval, in addition to or replacing the visibility check
        if rightHand.time_visible > 1:    #check if hand is still mobile/likely to not be someone trying to hold a sign
            if globletter != 0:
                newHand = handtoMatrix(rightHand)
                #print(newHand)
                handArray = newHand.flatten()#to get a list to put into the neural network  
                handArray = np.append(handArray, globletter)
                handDataFrame = pd.DataFrame(np.reshape(handArray, (1,len(handArray))))
                
                filename = "templates/test.csv" #build directory and file name
                filehandle = open(filename, 'a')    #open in append mode
                #cols = [palmx,palmy,palmz,thumb0x,thumb0y,thumb0z,thumb1x,thumb1y,thumb1z,thumb2x,thumb2y,thumb2z,thumb3x,thumb3y,thumn3z,index0x,index0y,index0z,index1x,index1y,index1z,index2x,index2y,index2z,index3x,index3y,index3z,middle0x,middle0y,middle0z,middle1x,middle1y,middle1z,middle2x,middle2y,middle2z,middle3x,middle3y,middle3z,ring0x,ring0y,ring0z,ring1x,ring1y,ring1z,ring2x,ring2y,ring2z,ring3x,ring3y,ring3z,pinky0x,pinky0y,pinky0z,pinky1x,pinky1y,pinky1z,pinky2x,pinky2y,pinky2z,pinky3x,pinky3y,pinky3z,sign]
                handDataFrame.to_csv(filehandle, header=False, index = False)
                globletter = 0
                print("saved "+filename)
                


def handtoMatrix(hand):
    handMatrix = np.zeros(shape=(21, 3))
    handMatrix[0, 0] = hand.palm_normal.x
    handMatrix[0, 1] = hand.palm_normal.y
    handMatrix[0, 2] = hand.palm_normal.z
    j = 1
    for finger in hand.fingers:
        for i in range(0, 4):
            bone = finger.bone(i)
            normBone = [bone.direction.x, bone.direction.y, bone.direction.z]
            if (bone.direction.x + bone.direction.y + bone.direction.z) != 0:   #since the first bone of the hand will always be, as they added one to the thumb
                normBone -= handMatrix[0,:]
                normBone = normalizeVector(normBone)
            handMatrix[j, :] = normBone
            j = j + 1
    return handMatrix

def normalizeVector(vector): #normalize vector given in list format.
    total = 0
    for num in vector:
        total = total+num*num
        #print(num)
    #print(total)
    magnitude = math.sqrt(total)
    if magnitude != 0:
        for num in vector:
            num = num/magnitude
    return vector

def main():
    listener = HandListener()
    controller = Leap.Controller()
    controller.add_listener(listener)
    global globletter
    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    getch = _Getch()
    loop = True;
    while loop:
        temp = getch()
        if temp != '\r' :
            globletter = temp.decode()
        else:
            loop = False
        
    
    controller.remove_listener(listener)

if __name__ == "__main__":
    main()