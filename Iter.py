import cv2
import numpy as np

def Iter1(image):

    r,c=image.shape
    print('r:',r,'c:',c)

    r_new,c_new=int(r/2),int(c/2)
    print('r_new:',r_new,'c_new:',c_new)
    
    while r_new>20 and c_new>20:

        image1=image[0:r_new,0:c_new]
        image2=image[0:r_new,c_new:c]
        image3=image[r_new:r,0:c_new]
        image4=image[r_new:r,c_new:c]

        cv2.namedWindow("1")
        cv2.namedWindow("2")
        cv2.namedWindow("3")
        cv2.namedWindow("4")

        cv2.imshow("1",image1)
        cv2.imshow("2",image2)
        cv2.imshow("3",image3)
        cv2.imshow("4",image4)
        cv2.waitKey(2000)

        Iter1(image1)
        Iter1(image2)
        Iter1(image3)
        Iter1(image4)
        return 


def fun1(i):
    print('f(',i,')')
    i+=1
    print(i)
    if i<5:
        return fun(i)
    else:
        return i

def fun2(i):
    print(i)
    if i/2>1:
        re=fun2(i/2)
        print('return value:',re)
    print('uppper value:',i)
    return i



if __name__=="__main__":
    img=cv2.imread("test/MgO.png",cv2.IMREAD_GRAYSCALE)
    #img=cv2.imread("test/104.jpg",cv2.IMREAD_GRAYSCALE)
    Iter1(img)
    #print(fun1(0))
    #print(fun2(10))
