import cv2
import numpy as np
 
image = cv2.imread('output.png')
height, width, channel = image.shape[:]
new_image = np.zeros((height, width, channel), np.uint8)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,50,apertureSize=3)
 
lines_list =[]
lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=200, # Min allowed length of line
            maxLineGap=500 # Max allowed gap between line for joining them
            )
 
for points in lines:
    x1,y1,x2,y2=points[0]
    cv2.line(new_image,(x1,y1),(x2,y2),(0,255,0),2)
    lines_list.append([(x1,y1),(x2,y2)])

new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(new_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(new_image, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)

cv2.imwrite('detectedLines.png',new_image)



# import cv2
# import numpy as np

# def process_image_for_edges(image_path, low_threshold=50, high_threshold=150):
#     # Read image
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
#     edges = cv2.Canny(blurred_img, low_threshold, high_threshold)

#     lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
    
#     # Draw lines on original image
#     line_img = np.copy(img)
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

#     return line_img

# if __name__ == '__main__':

#     processed_img = process_image_for_edges('output.png')
#     cv2.imshow('Processed Image', processed_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# import cv2
# import numpy as np

# def process_image_for_edges(image_path, low_threshold=1, high_threshold=150):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     edges = cv2.Canny(img,50,150,apertureSize = 3)

        
#     lines = cv2.HoughLines(edges,1,np.pi/180,200)
#     for line in lines:
#         rho,theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


#     return line_img

# if __name__ == '__main__':
#     processed_img = process_image_for_edges('output.png')
#     cv2.imshow('Processed Image', processed_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
