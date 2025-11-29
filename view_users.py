import sqlite3

# Connect to the database
conn = sqlite3.connect("users.db")
c = conn.cursor()

# Get all users
c.execute("SELECT * FROM users")
users = c.fetchall()

# Print each user
for u in users:
    print(u)

# Close connection
conn.close()
