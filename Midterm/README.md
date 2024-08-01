# Midtermd
kiem tra chieu dai chieu cao cua khung chu nhat 

Tach rieng vung xu ly -> luu vao 1 bien jpg roi xu ly. Co the tach ra thanh 4 vung. 
Dung opening to remove noise 

<!-- #Lab4. Q3
# img = cv2.imread('4_digits.png')
# img_gray = cv2.imread('4_digits.png',0)

# th1 = cv2.adaptiveThreshold(...)
# Invert color: black to white, white to black 
	# by	 bitwise_not(...) or use threhold(THRESH_BINARY_INV)
	
# img_lotNoise = th1[...] #Extract the region which have a lot of noise


# Opening, Closing

# contours, hierarchy  = cv2.findContours(...)
# for cnt in contours:
    # if(len(cnt) > < ...):
        # x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(...)
             
cv2.waitKey(0)
cv2.destroyAllWindows() -->

Chapter 1: nhớ ghi ra từng bước là 
- Mỗi bước chèn kết quả từng bước chạy  7

- tách ra 3 bước ảnh 
- gồm 4: top left, top right
        bottom left, bottom right 

## Note 
- Để sửa lại các tấm ảnh fix lại. 
Với 1 cái mình sẽ xử lý như sau:
- top left: đang dính 2 con số 2 thì mình cần opening mạnh hơn tí - DONE
- top right: đang dính 1 con số 2 thì mình cần set lại cái rectangle to lên 1 tí - DONE
- bottom left: đang dính con số 4 cần opening mạnh hơn. và con số 5. 
- bottom right: mình sẽ xử lý cuối cùng mình chưa thử dùng opening với closing mạnh.   Mình đang thấy nó hơi sao sao và con số 9 ở hàng 2


# Final 
Lấy 15 biển báo cấm có hình nền 
- Tối thiểu có 5 
- Xác định đừng viền -> Vị trị của biển báo cấm (đường tron có chu vi hay diện tích lớn nhất- mặt nạ, đường viền) -> Lọc màu để để lấy ra ()
- Ghi ra nội - Có 15 hình thì mình sẽ có nhãn 

- Có thể làm lab 6d để làm final và lab 9. 