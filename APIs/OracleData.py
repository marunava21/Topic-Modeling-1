import pandas as pd
from sqlalchemy import create_engine
class OracleData:
    def __init__(self, connection_String, username, password, hostname, port, database):
        self.oracle_connection_string = connection_String
        self.engine = create_engine(self.oracle_connection_string.format(
        username=username,
        password = password,
        hostname = hostname,
        port = port,
        database = database,
        ))
    def Query(self, query):
        return pd.read_sql(query, self.engine)
    def save_data(self, df, name):
        return df.to_excel(name)
if __name__ == '__main__':
    connection = OracleData(connection_String, username, password, hostname, port, database)