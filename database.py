import sqlite3
import hashlib
import datetime
import MySQLdb
from flask import session
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import numpy as np
import os
import cv2
import pandas as pd
def db_connect():
    _conn = MySQLdb.connect(host="localhost", user="root",
                            passwd="Sunny@14", db="admins")
    c = _conn.cursor()

    return c, _conn

# -------------------------------register-----------------------------------------------------------------
def acc_reg(username,email,password):
    try:
        status=acc_loginact(username, password)
        if status==1:
            return 0
        c, conn = db_connect()
        print(username, password, email)
        j = c.execute("INSERT INTO admins (username, email, password) VALUES (%s, %s, %s)", 
               (username, email, password))
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))
    
     
# -------------------------------------Login --------------------------------------
def acc_loginact(email, password):
    return 1



if __name__ == "__main__":
    print(db_connect())
