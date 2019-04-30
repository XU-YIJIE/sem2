To achieve the detection of gesture of hand and restore is backproject image and corresponding matrix, you need to :
1, Find a room with an ideal illumination
2, Lauch the camshift.py program
3, When you get the image from your camera, press B to switch the image to backproject mode
4, Don't move your head, and you should notice that your face has been darkened and some parts in the image that has similar color to your face has been highlighted
5, Raise your hand to let the ellipse track your hand. In the meantime, press space
6, Now you should have a 224*224 image of backproject image of your hand named '224224.png', a corresponding compressed 16*16 image named '1616.png', and the matrix of 16*16 image named '01.csv' output in the same file of your code