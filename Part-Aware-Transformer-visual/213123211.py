import cv2
# 读取源图像
image = cv2.imread('/22085400520/TBDE_visual/111.jpg')
# 设置目标图像大小
new_width = int(image.shape[1] * 0.5)
new_height = int(image.shape[0] * 0.5)
target_size = (new_width, new_height)
# 调用cv2.resize()函数进行缩放操作
resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)