from tkinter import *
  
root = Tk() 
root.geometry("400x300") 
root.title(" Q&A ") 
  
def Take_input(): 
    INPUT = inputtxt.get("1.0", "end-1c")
    print(INPUT) 
    if(INPUT == "120"): 
        Output.insert(END, 'Correct') 
    else: 
        Output.insert(END, "Wrong answer") 
      
l = Label(text = 'Masukan Kalimat Bahasa Indonesia atau Malaysia') 
inputtxt = Text(root, height = 10, 
                width = 50, 
                bg = "white") 
  
Output = Text(root, height = 5,  
              width = 50,  
              bg = "light cyan") 
  
Display = Button(root, text ="Prediksi Bahasa", 
                 command = lambda:Take_input()) 
  
l.pack() 
inputtxt.pack() 
Display.pack() 
Output.pack() 
  
mainloop() 