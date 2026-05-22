import sqlite3

conn= sqlite3.connect('sales_db.db')
cursor= conn.cursor()

cursor.execute("""
SELECT * FROM orders
""")

rows = cursor.fetchall()

for row in rows:
    print(row)

conn.commit()

conn.close()