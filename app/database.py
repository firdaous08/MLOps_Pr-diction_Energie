from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, Float, String, JSON, DateTime
from sqlalchemy.sql import func

SQLALCHEMY_DATABASE_URL = "postgresql://postgres:admin@localhost/energy_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()



class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    input_data = Column(JSON)
    predicted_value = Column(Float)
    model_version = Column(String, default="1.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()