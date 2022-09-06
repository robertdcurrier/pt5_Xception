import cv2 as cv2

if __name__ == '__main__':
	# The function imread loads an
	# image from the specified file and returns it.
	original = cv2.imread("testing/original_image.png")

	# Making another copy of an image.
	img = original.copy()

	# The function namedWindow creates
	# a window that can be used as
	# a placeholder for images.
	cv2.namedWindow('GEEK')

	# The function imshow displays
	# an image in the specified window.
	cv2.imshow('GEEK', original)

	# createTrackbar(trackbarName,
	# windowName, value, count, onChange)
	# Brightness range -255 to 255
	cv2.createTrackbar('Brightness', 'GEEK',
					255, 2 * 255,
					BrightnessContrast)

	# Contrast range -127 to 127
	cv2.createTrackbar('Contrast', 'GEEK',
					127, 2 * 127,
					BrightnessContrast)

	BrightnessContrast(0)

# The function waitKey waits for
# a key event infinitely or for
# delay milliseconds, when it is positive.
cv2.waitKey(0)

