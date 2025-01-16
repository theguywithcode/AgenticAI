import sqlite3

# Define the database name
db_name = "your_database.db"

# Establish connection to the database
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

# Create tables: properties and tenants
cursor.execute("""
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    address TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS tenants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    property_id INTEGER NOT NULL,
    FOREIGN KEY (property_id) REFERENCES properties (id)
)
""")

# Insert sample data into the properties table
properties = [
    ("Property A", "123 Main Street"),
    ("Property B", "456 Elm Street"),
    ("Property C", "789 Maple Avenue")
]
cursor.executemany("INSERT INTO properties (name, address) VALUES (?, ?)", properties)

# Insert sample data into the tenants table
tenants = [
    ("John Doe", 30, 1),  # Lives in Property A
    ("Jane Smith", 28, 1),  # Lives in Property A
    ("Alice Johnson", 35, 2),  # Lives in Property B
    ("Bob Brown", 40, 2),  # Lives in Property B
    ("Charlie Davis", 50, 3)  # Lives in Property C
]
cursor.executemany("INSERT INTO tenants (name, age, property_id) VALUES (?, ?, ?)", tenants)

# Commit changes and close the connection
connection.commit()
connection.close()

print(f"Sample data inserted into {db_name}!")
