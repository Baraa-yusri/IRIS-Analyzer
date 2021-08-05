
#import modules

from tkinter import *
import os

# Designing window for registration

def admin_verify_screen():
    global admin_verify_window
    admin_verify_window = Toplevel(main_screen)
    admin_verify_window.title("Admin Log in")
    admin_verify_window.geometry("300x300")
    admin_verify_window.configure(bg='#34495E')
    
    global pin
    global pin_entry
    pin = StringVar()
    
    Label(admin_verify_window, text="Admin Verification", bg="#F39C12", width="300", height="2", font=("Calibri", 16)).pack()
    Label(admin_verify_window, text="",bg='#34495E').pack()

    Label(admin_verify_window, text="Please enter 4-digits PIN",bg='#34495E', fg='white', width="150", height="2", font=("Calibri",12,'underline')).pack()
    Label(admin_verify_window, text="",bg='#34495E').pack()
      
    #pin_lable = Label(admin_verify_window, text="PIN")
    #pin_lable.pack()
    pin_entry = Entry(admin_verify_window, textvariable=pin, show='*')
    pin_entry.pack()
    
    Label(admin_verify_window, text="",bg='#34495E').pack()
    
    # Need to change command = register_user !! after doing the admin screen 
    Button(admin_verify_window, text="Enter", width=10, height=1, bg="white", command = admin_verify).pack()

# Implementing event on login button 

def admin_verify():
    entered_pin = pin.get()
    pin_entry.delete(0, END)
    
    valid_pin = "0000"

    if entered_pin == valid_pin:
        admin_sucess()

    else:
        pin_not_recognised()

# Designing popup for login success

def admin_sucess():
    global admin_screen
    admin_screen = Toplevel(admin_verify_window)
    admin_screen.geometry("300x300")
    admin_screen.title("Admin") 
    admin_screen.configure(bg='#34495E')
    Label(admin_screen, text="Admin", bg="#F39C12", width="300", height="2", font=("Calibri", 16)).pack()
    Label(admin_screen, text="",bg='#34495E').pack()
 
    Label(admin_screen, text="Select Your Choice",bg='#34495E', fg='white', width="150", height="2", font=("Calibri",12,'underline')).pack()
    Label(admin_screen, text="",bg='#34495E').pack()
    Button(admin_screen, text="Add an image", height="2", bg="white", width="30", command = "").pack()
    Label(admin_screen, text="", bg='#34495E').pack()
    Button(admin_screen, text="Delete an image", height="2", bg="white", width="30", command="").pack()


# Designing popup for login invalid password

def pin_not_recognised():
    global incorrect_pin
    incorrect_pin = Toplevel(admin_verify_window)
    incorrect_pin.title("Wrong pin")
    incorrect_pin.configure(bg='#34495E')
    incorrect_pin.geometry("200x200")
    Label(incorrect_pin,text="Invalid pin",bg='#34495E', fg='white', width="150", height="2", font=("Calibri",12,'underline')).pack()
    Label(incorrect_pin, text="",bg='#34495E').pack()
    Button(incorrect_pin, text="Try Again", height="2", bg="white", width="30", command=delete_pin_not_recognised).pack()

# Deleting popups
def delete_admin_success():
    admin_screen.destroy()

def delete_pin_not_recognised():
    incorrect_pin.destroy()

        
# # Designing window for login 

# def login():
    # global login_screen
    # login_screen = Toplevel(main_screen)
    # login_screen.title("Login")
    # login_screen.geometry("300x250")
    # Label(login_screen, text="Please enter details below to login").pack()
    # Label(login_screen, text="").pack()

    # global username_verify
    # global password_verify

    # username_verify = StringVar()
    # password_verify = StringVar()

    # global username_login_entry
    # global password_login_entry

    # Label(login_screen, text="Username * ").pack()
    # username_login_entry = Entry(login_screen, textvariable=username_verify)
    # username_login_entry.pack()
    # Label(login_screen, text="").pack()
    # Label(login_screen, text="Password * ").pack()
    # password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    # password_login_entry.pack()
    # Label(login_screen, text="").pack()
    # Button(login_screen, text="Login", width=10, height=1, command = login_verify).pack()

# # Implementing event on register button

# def register_user():

    # username_info = username.get()
    # password_info = password.get()

    # file = open(username_info, "w")
    # file.write(username_info + "\n")
    # file.write(password_info)
    # file.close()

    # username_entry.delete(0, END)
    # password_entry.delete(0, END)

    # Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack()



# Designing Main(first) window

def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("500x400")
    main_screen.configure(bg='#34495E')
    main_screen.title("Welcome to the IRIS ANALYSER Desktop App")
    Label(text="Iris Analyzer", bg="#F39C12", width="300", height="2", font=("Calibri", 16)).pack()
    Label(text="",bg='#34495E').pack()
    Label(text="Select Your Choice",bg='#34495E', fg='white', width="150", height="2", font=("Calibri",12,'underline')).pack()
    Label(text="",bg='#34495E').pack()
    Button(text="User", height="2", bg="white", width="30", command = "").pack()
    Label(text="", bg='#34495E').pack()
    Button(text="Admin", height="2", bg="white", width="30", command=admin_verify_screen).pack()

    main_screen.mainloop()


main_account_screen()
