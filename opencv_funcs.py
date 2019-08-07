import cv2
import time
import tempfile

esc_button = 27

def take_and_return_images(amount, camera):

    img_list = []
    for i in range(amount):
        ret_val, img = camera.read()
        img_list.append(img)

    return img_list


def show_camera_live():
    pic_num = 1
    spacebar = 32
    esc_button = 27
    path = "C:\\Users\\Tobia\\Pictures\\Camera Roll\\"
    window_name = "Definitely not spying"

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cam.isOpened():
        raise Exception("Could not open video device")

    while True:
        ret_val, img = cam.read()
        cv2.resize(img, (1280, 720))
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.imshow(window_name, img)

        if cv2.waitKey(1) == esc_button:
            break  # esc to quit
        if cv2.waitKey(1) == spacebar:
            cv2.imwrite(path + "test" + str(pic_num) + ".png", img)  # save a picture
            print("picture saved in " + path )
            pic_num += 1

    cv2.destroyAllWindows()
    del cam


def take_and_show_image():
    window_name = "Definitely not spying"
    cam = cv2.VideoCapture(0)
    ret_val, img = cam.read()

    while True:
        cv2.imshow(window_name, img)

        if cv2.waitKey(1) == esc_button:
            break

    cv2.destroyAllWindows()
    del cam


def take_and_save_images(img_directory, pic_amount, show_image=False):
    window_name = "Hey there, it's Big Brother"
    cam = cv2.VideoCapture(0)
    current_pics_taken = 0

    while True:
        ret_val, img = cam.read()
        take_pic = True

        if take_pic and not current_pics_taken == pic_amount:
            if show_image:
                cv2.imshow(window_name + str(current_pics_taken), img)
            cv2.imwrite(img_directory + "\\pic_" + str(current_pics_taken) +".jpg", img)
            take_pic = False
            current_pics_taken += 1
        if current_pics_taken == pic_amount or cv2.waitKey(1) == esc_button:
            break
    cv2.destroyAllWindows()
    del cam


def calculate_frame_rate():
    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    # Number of frames to capture
    num_frames = 100

    print("Capturing {0} frames".format(num_frames))

    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = cam.read()

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

    # Release video
    cam.release()
    cv2.destroyAllWindows()
    del cam


if __name__ == '__main__':
    #calculate_frame_rate()
    take_and_save_images()
    #take_and_show_image()
    #show_camera_live()
