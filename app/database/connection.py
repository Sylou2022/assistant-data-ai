import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def get_connection_string() -> str:
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    driver = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    trusted_connection = os.getenv(
        "SQL_TRUSTED_CONNECTION",
        "no",                   #.......................................
    )
    trust_server_certificate = os.getenv(
        "SQL_TRUST_SERVER_CERTIFICATE",
        "yes",
    )

    if not server:
        raise ValueError("La variable SQL_SERVER est absente du fichier .env")

    if not database:
        raise ValueError("La variable SQL_DATABASE est absente du fichier .env")

    if trusted_connection.lower() == "yes":
        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
            "Encrypt=yes;"
            f"TrustServerCertificate={trust_server_certificate};"
            "ApplicationIntent=ReadOnly;"
        )
    else:
        if not username or not password:
            raise ValueError(
                "SQL_USERNAME et SQL_PASSWORD sont nécessaires "
                "pour une authentification SQL Server."
            )

        connection_string = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            "Encrypt=no;"
            f"TrustServerCertificate={trust_server_certificate};"
        )

    return connection_string


def get_engine() -> Engine:
    connection_string = get_connection_string()
    encoded_connection_string = quote_plus(connection_string)

    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={encoded_connection_string}",
        pool_pre_ping=True,
        future=True,
    )


def test_connection() -> bool:
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return result.scalar() == 1