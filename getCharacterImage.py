"""
dans cette partie nous allons effectuez des test pour mesurer les caracteristique de notre dataset
"""
import cv2


img = cv2.imread("imageTest/imgBeans11.jpg")
print(img.shape)
imgBlur = cv2.GaussianBlur(img,(7,7),7)
imgCanny = cv2.Canny(imgBlur, 100, 100)

cv2.imshow("original", img)
cv2.imshow("Blur", imgBlur)
cv2.imshow("shape", imgCanny)
cv2.waitKey(0)