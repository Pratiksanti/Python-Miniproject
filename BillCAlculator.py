from tkinter import*
import math,random,os
from tkinter import messagebox
import re
class Bill_App:
    def __init__(self,root):
      self.root=root
      self.root.geometry("1350x700+0+0")
      self.root.title("Bill Generator")
      bg_color="#074463"
      title=Label(self.root,text="Billing Software",bd=12,relief=GROOVE,bg=bg_color,fg="white",font=("times new roman",30,"bold"),pady=2).pack(fill=X)

      ###   Variable  ###
      ###  Cosmetic   ###
      self.soap=IntVar()
      self.face_cream=IntVar()
      self.face_wash=IntVar()
      self.spray=IntVar()
      self.gell=IntVar()
      self.loshan=IntVar()

      ###  Grocery  ###
      self.rice=IntVar()
      self.food_oil=IntVar()
      self.daal=IntVar()
      self.wheat=IntVar()
      self.sugar=IntVar()
      self.tea=IntVar()

      ### Cold Drink  ###
      self.maza=IntVar()
      self.cock=IntVar()
      self.frooti=IntVar()
      self.thumbsup=IntVar()
      self.limca=IntVar()
      self.sprite=IntVar()

      ###  Total Product Price & Tax Variable
      self.cosmetic_price=StringVar()
      self.grocery_price=StringVar()
      self.cold_drink_price=StringVar()

      self.cosmetic_tax=StringVar()
      self.grocery_tax=StringVar()
      self.cold_drink_tax=StringVar()

      ###  Customer ###
      self.c_name=StringVar()
      self.c_phon=StringVar()
      self.bill_no=StringVar()
      x=random.randint(1000,9999)
      self.bill_no.set(str(x))
      self.search_bill=StringVar()


     # Customer Detail Frame
      f1 = LabelFrame(self.root, text="Customer Details", font=("times new roman", 15, "bold"), fg="gold", bg=bg_color)
      f1.place(x=0, y=80, relwidth=1)
     # Customer Name Label and Entry
      cname_lb1 = Label(f1, text="Customer Name", bg=bg_color, fg="white", font=("times new roman", 18, "bold")).grid(row=0, column=0, padx=20, pady=5)
      cname_txt = Entry(f1, width=20, textvariable=self.c_name, font="arial 15", bd=7, relief=SUNKEN, validate="key", validatecommand=(self.root.register(self.validate_name), '%P')).grid(row=0, column=1, pady=5, padx=10)


      # Phone Number Label and Entry
      cphn_lb1 = Label(f1, text="Phone No.", bg=bg_color, fg="white", font=("times new roman", 18, "bold")).grid(row=0, column=2, padx=20, pady=5)
      cphn_txt = Entry(f1, width=20, textvariable=self.c_phon, font="arial 15", bd=7, relief=SUNKEN, validate="key", validatecommand=(self.root.register(self.validate_phone), '%P')).grid(row=0, column=3, pady=5, padx=10)

      # Bill Number Label and Entry
      c_bill_lb1 = Label(f1, text="Bill Number", bg=bg_color, fg="white", font=("times new roman", 18, "bold")).grid(row=0, column=4, padx=20, pady=5)
      c_bill_txt = Entry(f1, width=20, textvariable=self.search_bill, font="arial 15", bd=7, relief=SUNKEN, validate="key", validatecommand=(self.root.register(self.validate_bill_number), '%P')).grid(row=0, column=5, pady=5, padx=10)

        # Search Button
      bill_btn = Button(f1, text="Search", command=self.find_bill, width=10, bd=7, font="arial 12 bold").grid(row=0, column=6, pady=10, padx=10)

       #Cosmatic Frame

      f2=LabelFrame(self.root,text="Cosmetics",font=("times new roman",15,"bold"),fg="gold",bg=bg_color)
      f2.place(x=5,y=180,width=325,height=380)

      bath_lb1=Label(f2,text="Bath Soap",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=0,column=0,padx=10,pady=10,sticky="w")
      bath_txt=Entry(f2,width=10,textvariable=self.soap,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=0,column=1,padx=10,pady=10)

      Face_cream_lb1=Label(f2,text="Face Cream",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=1,column=0,padx=10,pady=10,sticky="w")
      Face_cream_txt=Entry(f2,width=10,textvariable=self.face_cream,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=1,column=1,padx=10,pady=10)

      Face_w_lb1=Label(f2,text="Face Wash",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=2,column=0,padx=10,pady=10,sticky="w")
      Face_w_txt=Entry(f2,width=10,textvariable=self.face_wash,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=2,column=1,padx=10,pady=10)

      Hair_s_lb1=Label(f2,text="Hair Spray",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=3,column=0,padx=10,pady=10,sticky="w")
      Hair_s_txt=Entry(f2,width=10,textvariable=self.spray,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=3,column=1,padx=10,pady=10)

      Hair_g_lb1=Label(f2,text="Hair Gell",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=4,column=0,padx=10,pady=10,sticky="w")
      Hair_g_txt=Entry(f2,width=10,textvariable=self.gell,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=4,column=1,padx=10,pady=10)

      body_lb1=Label(f2,text="Body Loshan ",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=5,column=0,padx=10,pady=10,sticky="w")
      body_txt=Entry(f2,width=10,textvariable=self.loshan,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=5,column=1,padx=10,pady=10)


       #Glocery Frame

      f3=LabelFrame(self.root,text="Glosary Product",font=("times new roman",15,"bold"),fg="gold",bg=bg_color)
      f3.place(x=360,y=180,width=325,height=380)

      g1_lb1=Label(f3,text="Rice",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=0,column=0,padx=10,pady=10,sticky="w")
      g1_txt=Entry(f3,width=10,textvariable=self.rice,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=0,column=1,padx=10,pady=10)

      g2_lb1=Label(f3,text="Food Oil",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=1,column=0,padx=10,pady=10,sticky="w")
      g2_cream_txt=Entry(f3,width=10,textvariable=self.food_oil,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=1,column=1,padx=10,pady=10)

      g3_lb1=Label(f3,text="Daal",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=2,column=0,padx=10,pady=10,sticky="w")
      g3_txt=Entry(f3,width=10,textvariable=self.daal,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=2,column=1,padx=10,pady=10)

      g4_lb1=Label(f3,text="Wheat",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=3,column=0,padx=10,pady=10,sticky="w")
      g4_txt=Entry(f3,width=10,textvariable=self.wheat,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=3,column=1,padx=10,pady=10)

      g5_lb1=Label(f3,text="Sugar",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=4,column=0,padx=10,pady=10,sticky="w")
      g5_txt=Entry(f3,width=10,textvariable=self.sugar,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=4,column=1,padx=10,pady=10)

      g6_lb1=Label(f3,text="Tea",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=5,column=0,padx=10,pady=10,sticky="w")
      g6_txt=Entry(f3,width=10,textvariable=self.tea,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=5,column=1,padx=10,pady=10)


      
       #Cold Drink Frame

      f4=LabelFrame(self.root,text="Cold Drink",font=("times new roman",15,"bold"),fg="gold",bg=bg_color)
      f4.place(x=715,y=180,width=325,height=380)

      c1_lb1=Label(f4,text="Maza",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=0,column=0,padx=10,pady=10,sticky="w")
      c1_txt=Entry(f4,width=10,textvariable=self.maza,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=0,column=1,padx=10,pady=10)

      c2_lb1=Label(f4,text="Cock",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=1,column=0,padx=10,pady=10,sticky="w")
      c2_cream_txt=Entry(f4,width=10,textvariable=self.cock,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=1,column=1,padx=10,pady=10)

      c3_lb1=Label(f4,text="Frooti",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=2,column=0,padx=10,pady=10,sticky="w")
      c3_txt=Entry(f4,width=10,textvariable=self.frooti,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=2,column=1,padx=10,pady=10)

      c4_lb1=Label(f4,text="Thumbs Up",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=3,column=0,padx=10,pady=10,sticky="w")
      c4_txt=Entry(f4,width=10,textvariable=self.thumbsup,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=3,column=1,padx=10,pady=10)

      c5_lb1=Label(f4,text="Limca",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=4,column=0,padx=10,pady=10,sticky="w")
      c5_txt=Entry(f4,width=10,textvariable=self.limca,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=4,column=1,padx=10,pady=10)

      c6_lb1=Label(f4,text="Sprite",font=("times new roman",16,"bold"),bg=bg_color,fg="lightgreen").grid(row=5,column=0,padx=10,pady=10,sticky="w")
      c6_txt=Entry(f4,width=10,textvariable=self.sprite,font=("times new roman",16,"bold"),bd=5,relief=SUNKEN).grid(row=5,column=1,padx=10,pady=10)


      #Bill Area

      f5=LabelFrame(self.root,bd=10)
      f5.place(x=1100,y=180,width=385,height=380)

      bill_title=Label(f5,text="Bill Area",font="arial 15 bold",bd=7,relief=GROOVE).pack(fill=X)
      scrol_y=Scrollbar(f5,orient=VERTICAL)
      self.txtarea= Text(f5,yscrollcommand=scrol_y.set)
      scrol_y.pack(side=RIGHT,fill=Y)
      scrol_y.config(command=self.txtarea.yview)
      self.txtarea.pack()


      #Button Frame   3rd layer

      f6=LabelFrame(self.root,text="Bill Menu",font=("times new roman",15,"bold"),fg="gold",bg=bg_color)
      f6.place(x=0,y=570,relwidth=1,height=200)
      m1_lb1=Label(f6,text="Total Cosmetic Price",bg=bg_color,fg="white",font=("times new roman",14,"bold")).grid(row=0,column=0,padx=20,pady=1,sticky="w")
      m1_txt=Entry(f6,width=18,textvariable=self.cosmetic_price,font="arial 10 bold",bd=7,relief=SUNKEN).grid(row=0,column=1,padx=10,pady=10)

      m2_lb1=Label(f6,text="Total Grocery Price",bg=bg_color,fg="white",font=("times new roman",14,"bold")).grid(row=1,column=0,padx=20,pady=1,sticky="w")
      m2_txt=Entry(f6,width=18,textvariable=self.grocery_price,font="arial 10 bold",bd=7,relief=SUNKEN).grid(row=1,column=1,padx=10,pady=10)

      m3_lb1=Label(f6,text="Total Cold Drinks Price",bg=bg_color,fg="white",font=("times new roman",14,"bold")).grid(row=2,column=0,padx=20,pady=1,sticky="w")
      m3_txt=Entry(f6,width=18,textvariable=self.cold_drink_price,font="arial 10 bold",bd=7,relief=SUNKEN).grid(row=2,column=1,padx=10,pady=10)



      c1_lb1=Label(f6,text="Cosmetic Tax",bg=bg_color,fg="white",font=("times new roman",14,"bold")).grid(row=0,column=2,padx=20,pady=1,sticky="w")
      c1_txt=Entry(f6,width=18,textvariable=self.cosmetic_tax,font="arial 10 bold",bd=7,relief=SUNKEN).grid(row=0,column=3,padx=10,pady=1)

      c2_lb1=Label(f6,text="Grocery Tax",bg=bg_color,fg="white",font=("times new roman",14,"bold")).grid(row=1,column=2,padx=20,pady=1,sticky="w")
      c2_txt=Entry(f6,width=18,textvariable=self.grocery_tax,font="arial 10 bold",bd=7,relief=SUNKEN).grid(row=1,column=3,padx=10,pady=4)

      c3_lb1=Label(f6,text="Cold Drinks Tax",bg=bg_color,fg="white",font=("times new roman",14,"bold")).grid(row=2,column=2,padx=20,pady=1,sticky="w")
      c3_txt=Entry(f6,width=18,textvariable=self.cold_drink_tax,font="arial 10 bold",bd=7,relief=SUNKEN).grid(row=2,column=3,padx=10,pady=1)

      btn_F=Frame(f6,bd=7,relief=GROOVE)
      btn_F.place(x=850,width=650,height=150)

      total_btn=Button(btn_F,command=self.total,text="Total",bg="cadetblue",fg="white",pady=36,width=11,font="arial 15 bold",bd=5).grid(row=0,column=0,padx=5,pady=5)
      GBill_btn=Button(btn_F,text="Generate Bill",command=self.bill_area,bg="cadetblue",fg="white",pady=36,width=11,font="arial 15 bold",bd=5).grid(row=0,column=1,padx=5,pady=5)
      Clear_btn=Button(btn_F,text="Clear",command=self.clear_data,bg="cadetblue",fg="white",pady=36,width=11,font="arial 15 bold",bd=5).grid(row=0,column=2,padx=5,pady=5)
      Exit_btn=Button(btn_F,text="Exit",command=self.Exit_app,bg="cadetblue",fg="white",pady=36,width=11,font="arial 15 bold",bd=5).grid(row=0,column=3,padx=5,pady=5)
      self.welcome_bill()
     

     #validation for customer name
    def validate_name(self, name):
        """Only allow alphabetic characters and spaces in customer name."""
        return bool(re.match(r"^[A-Za-z\s]*$", name))
   # validation for phone number
    def validate_phone(self, phone):
        """Only allow numbers and limit length to 10 digits for phone number."""
        return bool(re.match(r"^\d{0,10}$", phone))
   # validation for bill_number
    def validate_bill_number(self, bill_number):
        """Only allow numeric bill numbers."""
        return bool(re.match(r"^\d*$", bill_number))

   # Dummy find_bill function (for the example)
    def find_bill(self):
        print("Searching bill...")

      #Functionalities

    def total(self):
      self.c_s_p=self.soap.get()*40
      self.c_fc_p=self.face_cream.get()*40
      self.c_fw_p=self.face_wash.get()*40
      self.c_hs_p=self.spray.get()*40
      self.c_hg_p=self.gell.get()*40
      self.c_bl_p=self.loshan.get()*40
      self.total_cosmetic_price=float(
                                      self.c_s_p+
                                      self.c_fc_p+
                                      self.c_fw_p+
                                      self.c_hs_p+
                                      self.c_hg_p+
                                      self.c_bl_p
                                 
      )

      self.cosmetic_price.set("Rs "+str(self.total_cosmetic_price))
      self.c_tax=round((self.total_cosmetic_price*0.05),2)
      self.cosmetic_tax.set("Rs "+str(self.c_tax))


      self.g_r_p=self.rice.get()*80
      self.g_f_p=self.food_oil.get()*180
      self.g_d_p=self.daal.get()*80
      self.g_w_p=self.wheat.get()*80
      self.g_s_p=self.sugar.get()*40
      self.g_t_p=self.tea.get()*80

      self.total_grocery_price=float(
                                  self.g_r_p+
                                  self.g_f_p+
                                  self.g_d_p+
                                  self.g_w_p+
                                  self.g_s_p+
                                  self.g_t_p
      )

      self.grocery_price.set("Rs "+str(self.total_grocery_price))
      self.g_tax=round((self.total_grocery_price*0.1),2)
      self.grocery_tax.set("Rs "+str(self.g_tax))

      self.d_m_p=self.maza.get()*60
      self.d_c_p=self.cock.get()*60
      self.d_f_p=self.frooti.get()*50
      self.d_t_p=self.thumbsup.get()*40
      self.d_l_p=self.limca.get()*60
      self.d_s_p=self.sprite.get()*40

      self.total_drink_price=float(
                                   self.d_m_p+
                                   self.d_c_p+
                                   self.d_f_p+
                                   self.d_t_p+
                                   self.d_l_p+
                                   self.d_s_p
      )

      self.cold_drink_price.set("Rs "+str(self.total_drink_price))
      self.d_tax=round((self.total_drink_price*0.05),2)
      self.cold_drink_tax.set("Rs "+str(self.d_tax))

      self.Total_bill=float(self.total_cosmetic_price
                              + self.total_grocery_price
                               + self.total_drink_price
                               +self.c_tax
                               +self.g_tax
                               +self.d_tax
      )

    def welcome_bill(self):
      self.txtarea.delete('1.0',END)
      self.txtarea.insert(END,"\t Welcome To Retail\n")
      self.txtarea.insert(END,f"\n Bill Number: {self.bill_no.get()}")
      self.txtarea.insert(END,f"\n Customer Name:{self.c_name.get()}")
      self.txtarea.insert(END,f"\n Phone Number: {self.c_phon.get()}")
      self.txtarea.insert(END,f"\n===========================================")
      self.txtarea.insert(END,f"\n Products\t\tQTY\t\tPrice")
      self.txtarea.insert(END,f"\n===========================================")
    def bill_area(self):
      if self.c_name.get()=="" or self.c_phon.get()=="":
        messagebox.showerror("Error","Customer details are must")
      elif self.cosmetic_price.get()=="Rs 0.0" and self.grocery_price.get()=="Rs 0.0" and self.cold_drink_price.get()=="Rs 0.0":
          messagebox.showerror("Error","No Product Purchased")
      else:
        self.welcome_bill()
        #Cosmetics Display on Bill area#
        if self.soap.get()!=0:
          self.txtarea.insert(END,f"\n Bath Soap\t\t{self.soap.get()}\t\t{self.c_s_p}")

        if self.face_cream.get()!=0:
          self.txtarea.insert(END,f"\n Face Cream \t\t{self.face_cream.get()}\t\t{self.c_fc_p}")

        if self.face_wash.get()!=0:
          self.txtarea.insert(END,f"\n Face Wash \t\t{self.face_wash.get()}\t\t{self.c_fw_p}")

        if self.spray.get()!=0:
          self.txtarea.insert(END,f"\n Spray \t\t{self.spray.get()}\t\t{self.c_hs_p}")

        if self.gell.get()!=0:
          self.txtarea.insert(END,f"\n Hair Gell \t\t{self.gell.get()}\t\t{self.c_hg_p}")

        if self.loshan.get()!=0:
          self.txtarea.insert(END,f"\n Body Loshan\t\t{self.loshan.get()}\t\t{self.c_bl_p}")


        #Glocery Display on Bill area#
        if self.rice.get()!=0:
          self.txtarea.insert(END,f"\n Rice\t\t{self.rice.get()}\t\t{self.g_r_p}")

        if self.food_oil.get()!=0:
          self.txtarea.insert(END,f"\n Food Oil \t\t{self.food_oil.get()}\t\t{self.g_f_p}")

        if self.daal.get()!=0:
          self.txtarea.insert(END,f"\n Daal \t\t{self.daal.get()}\t\t{self.g_d_p}")

        if self.wheat.get()!=0:
          self.txtarea.insert(END,f"\n Wheat \t\t{self.wheat.get()}\t\t{self.g_w_p}")

        if self.sugar.get()!=0:
          self.txtarea.insert(END,f"\n Sugar \t\t{self.sugar.get()}\t\t{self.g_s_p}")

        if self.tea.get()!=0:
          self.txtarea.insert(END,f"\n Tea \t\t{self.tea.get()}\t\t{self.g_t_p}")

        #Cold Drink Display on Bill area#
        if self.maza.get()!=0:
          self.txtarea.insert(END,f"\n Maza\t\t{self.maza.get()}\t\t{self.d_m_p}")

        if self.cock.get()!=0:
          self.txtarea.insert(END,f"\n Cock  \t\t{self.cock.get()}\t\t{self.d_c_p}")

        if self.frooti.get()!=0:
          self.txtarea.insert(END,f"\n Frooti \t\t{self.frooti.get()}\t\t{self.d_f_p}")

        if self.thumbsup.get()!=0:
          self.txtarea.insert(END,f"\n Thumbsup \t\t{self.thumbsup.get()}\t\t{self.d_t_p}")

        if self.limca.get()!=0:
          self.txtarea.insert(END,f"\n Limca \t\t{self.limca.get()}\t\t{self.d_l_p}")

        if self.sprite.get()!=0:
          self.txtarea.insert(END,f"\n Sprite \t\t{self.sprite.get()}\t\t{self.d_s_p}")

        self.txtarea.insert(END,f"\n-------------------------------------------")
        if self.cosmetic_tax.get()!="Rs 0.0":
          self.txtarea.insert(END,f"\n Cosmetic Tax\t\t\t\t{self.cosmetic_tax.get()}")
        
        if self.grocery_tax.get()!="Rs 0.0":
          self.txtarea.insert(END,f"\n Glocery Tax \t\t\t\t{self.grocery_tax.get()}")

        if self.cold_drink_tax.get()!="Rs 0.0":
          self.txtarea.insert(END,f"\n Cold Drink Tax\t\t\t\t{self.cold_drink_tax.get()}")
        self.txtarea.insert(END,f"\n Total Bill\t\t\t\t Rs. {self.Total_bill}")
        self.txtarea.insert(END,f"\n-------------------------------------------")
        self.save_bill()

    def save_bill(self):
      op=messagebox.askyesno("Save Bill","Do You Want To Save Bill ?")
      if op>0:
        self.bill_data=self.txtarea.get('1.0',END)
        f1=open("bills/"+str(self.bill_no.get())+".txt","w")
        f1.write(self.bill_data)
        f1.close()
        messagebox.showinfo("Saved",f"Bill no. :{self.bill_no.get()} Saved Successfully")
      else:
        return
      
    def find_bill(self):
      present="no"
      for i in os.listdir("bills/"):
        if i.split('.')[0]==self.search_bill.get():
          f1=open(f"bills/{i}","r")
          self.txtarea.delete('1.0',END)
          for d in f1:
            self.txtarea.insert(END,d)
          f1.close()
          present="yes"
      if present=="no":
        messagebox.showerror("Error","Invalid Bill No.")

    def clear_data(self):
       op=messagebox.askyesno("Exit","Do You Want To Clear")
       if op>0:
      ###  Cosmetic   ###
        self.soap.set(0)
        self.face_cream.set(0)
        self.face_wash.set(0)
        self.spray.set(0)
        self.gell.set(0)
        self.loshan.set(0)

        ###  Grocery  ###
        self.rice.set(0)
        self.food_oil.set(0)
        self.daal.set(0)
        self.wheat.set(0)
        self.sugar.set(0)
        self.tea.set(0)

        ### Cold Drink  ###
        self.maza.set(0)
        self.cock.set(0)
        self.frooti.set(0)
        self.thumbsup.set(0)
        self.limca.set(0)
        self.sprite.set(0)

        ###  Total Product Price & Tax Variable
        self.cosmetic_price.set("")
        self.grocery_price.set("")
        self.cold_drink_price.set("")

        self.cosmetic_tax.set("")
        self.grocery_tax.set("")
        self.cold_drink_tax.set("")

        ###  Customer ###
        self.c_name.set("")
        self.c_phon.set("")
        self.bill_no.set("")
        x=random.randint(1000,9999)
        self.bill_no.set(str(x))
        self.search_bill.set("")
        self.welcome_bill()

    def Exit_app(self):
      op=messagebox.askyesno("Exit","Do You Want To Exist")
      if op>0:
        self.root.destroy()
      
  


root = Tk()
obj = Bill_App(root)
root.mainloop()       



