import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog

image = np.ones((55, 55, 3), dtype=np.uint8) * 255
message = ""
final_message = ""
max_message_bits = 64
encrypt_path = ""
decrypt_path = ""
quality = 50
block_size = 8
delimiter = '##END'
delimiter = ''.join(format(ord(char), '08b') for char in delimiter)

Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])

DQ = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 41, 60, 55],
        [14, 13, 16, 17, 28, 40, 48, 56],
        [14, 17, 22, 20, 36, 61, 56, 43],
        [18, 22, 26, 39, 48, 76, 72, 54],
        [24, 25, 39, 45, 57, 73, 79, 64],
        [49, 64, 55, 61, 72, 85, 84, 71],
        [72, 92, 95, 69, 78, 70, 72, 70]])


def zero_length(set):
    length = 0
    for i in set:
        if (i == 0 or i == -0):
            length += 1
        else:
            return length
    return length

def highest_non_zero(set):
    index = -10000
    for i in range(len(set)):
        if (set[i] != 0):
            index = i
            return index
    return index

def get_blocks(image, do):
    num_blocks_height = image.shape[0] // block_size
    num_blocks_width = image.shape[1] // block_size
    blocks = []

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            
            
            dct_block = cv2.dct(np.float32(block))
            if (do):
                quantized_dct_block = np.round(dct_block / (DQ * (quality / 50)))
            else:            
                
                quantized_dct_block = np.round(dct_block / (Q * (quality / 50)))

            blocks.append(quantized_dct_block)
    return blocks

def get_sets_from_blocks(blocks): 
    sets = []
    for i in blocks:   
        set1 = []
        set2 = []
        set3 = []
        set4 = []
        set5 = []
        set6 = []
        set7 = []
        set8 = []
        set9 = []
        allSets = {}
        counter = 1
        key = "Set"

        for x in range(0,7):
            set1.append(i[x,x])
        set1.reverse()
        allSets[key+str(counter)] = set1
        counter += 1

        for x in range(0,7):
            set2.append(i[x,x+1])
        set2.reverse()
        allSets[key+str(counter)] = set2
        counter += 1

        for x in range(0,7):
            set3.append(i[x+1,x])
        set3.reverse()
        allSets[key+str(counter)] = set3
        counter += 1

        for x in range(0,6):
            set4.append(i[x,x+2])
        set4.reverse()
        allSets[key+str(counter)] = set4
        counter += 1

        for x in range(0,6):
            set5.append(i[x+2,x])
        set5.reverse()
        allSets[key+str(counter)] = set5
        counter += 1

        for x in range(0,5):
            set6.append(i[x,x+3])
        set6.reverse()
        allSets[key+str(counter)] = set6
        counter += 1

        for x in range(0,5):
            set7.append(i[x+3,x])
        set7.reverse()
        allSets[key+str(counter)] = set7
        counter += 1

        for x in range(0,4):
            set8.append(i[x,x+4])
        set8.reverse()
        allSets[key+str(counter)] = set8
        counter += 1

        for x in range(0,4):
            set9.append(i[x+4,x])
        set9.reverse()
        allSets[key+str(counter)] = set9
        counter += 1
        
        sets.append(allSets)

    return sets

def dequantize_blocks(blocks):
    dequantized_blocks = []

    
    for quantized_block in blocks:
        
        dequantized_block = quantized_block * (DQ * (quality / 50))
        
        
        dequantized_block = cv2.idct(np.float32(dequantized_block))
        
        
        dequantized_block = np.uint8(np.clip(dequantized_block, 0, 255))
        
        
        dequantized_blocks.append(dequantized_block)

    return dequantized_blocks

def get_image_from_blocks(blocks):
    num_blocks_height = image.shape[0] // block_size
    num_blocks_width = image.shape[1] // block_size

    
    image_height, image_width = num_blocks_height * block_size, num_blocks_width * block_size
    reconstructed_image = np.zeros((image_height, image_width), dtype=np.uint8)

    
    block_index = 0
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
    
            block = blocks[block_index]
            
            
            reconstructed_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block
            
            block_index += 1

    return reconstructed_image

def hide_message(sets, binary_array):
    set_array = []
    counter = 0
    counter2 = 0

    for i in sets:
        for j in i:
            set_array.append(i[j]) 
    
    for i in set_array:
        if (counter < len(binary_array)):
            last_zero = zero_length(i)
            
            
            if((i[last_zero - 1] == 1 or i[last_zero - 1] == -1) and i[last_zero] == 0):
                        if (i[last_zero - 1] > 0):
                            i[last_zero - 1] += 1
                        else:
                            i[last_zero - 1] -= 1 

            
            if ((i[0] == -1 or i[0] == 1) and i[1] == 0):
                if (i[0] > 0):
                    i[0] += 1
                else:
                    i[0] -= 1

            
            if ((i[1] == -1 or i[1] == 1) and i[0] == 0 and i[2] == 0):
                if (i[1] > 0):
                    i[1] += 1
                else:
                    i[1] -= 1
            
            if (last_zero >= 2):                   
                if (binary_array[counter] == 1):
                    secret = random.randint(0, 1)
                    i[last_zero - 2] = 1 if secret == 1 else -1
                else:
                    i[last_zero - 2] = 0 
                counter = counter + 1
        elif(counter < len(binary_array) + len(delimiter)):
            last_zero = zero_length(i)
            if (delimiter[counter2] == '1'):
                if(counter2 < len(delimiter) - 1):
                    secret = random.randint(0, 1)
                    i[last_zero - 2] = 1 if secret == 1 else -1
                    counter2 = counter2 + 1 
            else:
                if(counter2 < len(delimiter) - 1):
                    i[last_zero - 2] = 0 
                    counter2 = counter2 + 1 
    return set_array

def encrypt():
    global encrypt_path, image
    image = cv2.imread(encrypt_path, cv2.IMREAD_GRAYSCALE)
    height = image.shape[0]
    width = image.shape[1]

    if (height % 8 != 0):
        height = height + 8 - (height % 8)
    if (width % 8 != 0):
        width = width + 8 - (width % 8)
    image = cv2.resize(image, (width, height))
    cv2.imwrite(encrypt_path, image)
    blocks = get_blocks(image, 0)  
    sets = get_sets_from_blocks(blocks)
    
    counter = 0

    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_array = [int(bit) for bit in binary_message]
    msg_length = len(binary_array)
    msg_length_bin = [int(bit) for bit in bin(msg_length)[2:]]
    
    set_array = hide_message(sets, binary_array)

    for i in blocks:
        set_array[counter].reverse()
        for x in range(0,7):
            i[x,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,7):
            i[x,x+1] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,7):
            i[x+1,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,6):
            i[x,x+2] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,6):
            i[x+2,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,5):
            i[x,x+3] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,5):
            i[x+3,x] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,4):
            i[x,x+4] = set_array[counter][x]
        counter = counter + 1

        set_array[counter].reverse()
        for x in range(0,4):
            i[x+4,x] = set_array[counter][x]
        counter = counter + 1  

    dequantized_blocks = dequantize_blocks(blocks)
    image_with_info = get_image_from_blocks(dequantized_blocks)
    dct_image_with_info = cv2.dct(np.float32(image_with_info))

    save_path = filedialog.asksaveasfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])

    cv2.imshow('hidden', image_with_info)
    cv2.imwrite(save_path, image_with_info)


    cv2.waitKey(0)

    cv2.destroyAllWindows()

def get_message(sets):
    received_message = ""
    received_length_bin = ""
    set_array = []

    for i in sets:
        for j in i:
            set_array.append(i[j])
            
    z = 0
    for i in set_array:    
        if (received_message[-40:] != delimiter):
            index = highest_non_zero(i)
            if (index != -10000):
                if (index < len(i) - 1):            
                    if ((i[index] == 1 or i[index] == -1) and i[index + 1] == 0):
                        received_message += "1"
                        z += 1    
                    elif (i[index - 1] == 0):
                        received_message += "0"
                        z += 1                  
                else:
                    if (not ((i[index - 1] == 1 or i[index - 1] == -1) and i[index] != 0) or (i[index - 1] != 1 and i[index - 1] != -1 and index <= 1)):      
                        received_message += "0"
                        z += 1
            else:               
                received_message += "0"
                z += 1               
        else:
            break
    return received_message[:-40]

def decrypt():
    global final_message, decrypt_path
    hidden = cv2.imread(decrypt_path, cv2.IMREAD_GRAYSCALE)
    blocks = get_blocks(hidden, 1)
    sets = get_sets_from_blocks(blocks)
    

    recived_message = get_message(sets)
    
    final_message = ''.join(chr(int(recived_message[i:i+8], 2)) for i in range(0, len(recived_message), 8))
    

root = tk.Tk()
root.title("DCT Steganografi Agil, Rian")


input_label = tk.Label(root, text="Teks yang ingin disembunyikan : ")
input_label.grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)

output_label = tk.Label(root, text="Hasil Ekstraksi : ")
output_label.grid(row=1, column=0, padx=3, pady=3, sticky=tk.W)


input_entry = tk.Text(root, height=5, width=50)
input_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)


output_entry = tk.Text(root, height=5, width=50)
output_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
output_entry.config(state=tk.DISABLED) 

def select_encrypt():
    global encrypt_path
    encrypt_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])

def select_decrypt():
    global decrypt_path
    decrypt_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])

def encrypt_file():
    global input_entry, message
    message = input_entry.get("1.0", tk.END)
    encrypt()
        
def decrypt_file():
    global output_entry, final_message
    decrypt()
    output_entry.config(state=tk.NORMAL) 
    output_entry.delete(1.0, tk.END)
    output_entry.insert(tk.END, final_message)
    output_entry.config(state=tk.DISABLED) 
        
import_encrypt = tk.Button(root, text="Pilih Gambar yang akan disisipi", command=select_encrypt)
import_decrypt = tk.Button(root, text="Pilih Gambar yang akan diekstrak", command=select_decrypt)
import_encrypt.grid(row=3, column=0, padx=5, pady=5, sticky=tk.SW)
import_decrypt.grid(row=3, column=1, padx=5, pady=5, sticky=tk.SE)

enc = tk.Button(root, text="Enkripsi", command=encrypt_file)
dec = tk.Button(root, text="Dekripsi", command=decrypt_file)
enc.grid(row=4, column=0, padx=5, pady=5)
dec.grid(row=4, column=1, padx=5, pady=5)

root.mainloop()




