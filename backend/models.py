from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    before_filename = Column(String)
    after_filename = Column(String)
    result_filename = Column(String)
    mask_filename = Column(String)
    added_count = Column(Integer, default=0)
    removed_count = Column(Integer, default=0)
    moved_count = Column(Integer, default=0)
    caption_mode = Column(String, default="pattern")
    caption = Column(Text)
    full_result = Column(Text)  # JSON blob
