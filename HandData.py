import os, sys
import Leap
import numpy as np
import math

class HandListener(Leap.Listener):

    def on_connect(self, controller):
        print("Connected")


    def on_frame(self, controller): #every time a frame is possible, this updates
        #print("Frame available")
        frame = controller.frame()
        rightHand = frame.hands.rightmost
        #print(rightHand)

        #add a timer which prevents checking the hand until the next interval, in addition to this
        if rightHand.time_visible > 1:    #check if hand is still mobile/likely to not be someone trying to hold a sign
            newHand = normalizeHandtoMatrix(rightHand)
            #print(newHand)
            handArray = newHand.flatten()#to get a list to put into the neural network
            print handArray
            


def normalizeHandtoMatrix(hand):
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
                normBone = normalizeVector(normBone)
            handMatrix[j, :] = normBone
            j = j + 1
    return handMatrix

def normalizeVector(vector):
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
    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)

if __name__ == "__main__":
    main()