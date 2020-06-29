# https://github.com/Suke-H/8point_algorithm
import cv2
import time


class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):

        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent

    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])


if __name__ == "__main__":
    #入力画像
    # img = cv2.imread("images/atomic_rapid_analyze.jpg")
    img = cv2.imread("images/hotel_chromosome_probe.jpg")
    # img = cv2.imread("images/new_citizen_master.jpg")
    # img = cv2.imread("images/rage_sticky_contestant.jpg")

    orgHeight, orgWidth = img.shape[:2]
    size = (int(orgHeight/3), int(orgWidth/3))
    print(size)
    img = cv2.resize(img, size[::-1])

    cv2.imwrite("img/resize2.jpg", img)

    #表示するWindow名
    window_name = "input window"

    #画像の表示
    cv2.imshow(window_name, img)

    #コールバックの設定
    mouseData = mouseParam(window_name)

    while 1:
        cv2.waitKey(5)
        #左クリックがあったら表示
        if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
            x, y = [i * 3 for i in mouseData.getPos()]
            print(f"    - [{x}, {y}]")
            time.sleep(0.2)
        #右クリックがあったら終了
        elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
            break

    cv2.destroyAllWindows()
    print("Finished")
