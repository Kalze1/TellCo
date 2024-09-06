import psycopg2
import pandas as pd 
import os 



#Replace with your database credentials
host = os.environ.get('DATABASE_HOST')
database = os.environ.get('DATABASE_NAME')
user = os.environ.get('DATABASE_USER')
password = os.environ.get('DATABASE_PASSWORD')

def load_df():

    conn = psycopg2.connect(
        database = "tellco",
        user = "postgres",
        password = "youwork"
    )

    cursor = conn.cursor()

    query = "SELECT * FROM public.xdr_data"
    cursor.execute(query)

    results = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description ]
    df = pd.DataFrame(results, columns=columns)

    cursor.close()
    

    return df



