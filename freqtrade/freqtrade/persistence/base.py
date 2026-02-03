from sqlalchemy.orm import DeclarativeBase, Session, scoped_session


SessionType = scoped_session[Session]


class ModelBase(DeclarativeBase):
    pass
