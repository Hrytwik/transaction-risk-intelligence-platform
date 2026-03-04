import mysql.connector

def get_connection():
    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",          # or your MySQL user
        password="Human1after'all",
        database="anomaly_detection",
    )
    return conn
