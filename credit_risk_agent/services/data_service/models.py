from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ClientDB(Base):
    __tablename__ = "clients"

    client_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    limit_bal: Mapped[float] = mapped_column(Float, nullable=False)
    sex: Mapped[int] = mapped_column(Integer, nullable=False)
    education: Mapped[int] = mapped_column(Integer, nullable=False)
    marriage: Mapped[int] = mapped_column(Integer, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)


class PaymentHistoryDB(Base):
    __tablename__ = "payment_history"

    client_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clients.client_id", ondelete="CASCADE"), primary_key=True
    )
    month: Mapped[int] = mapped_column(Integer, primary_key=True)
    pay_status: Mapped[int] = mapped_column(Integer, nullable=False)
    bill_amt: Mapped[float] = mapped_column(Float, nullable=False)
    pay_amt: Mapped[float] = mapped_column(Float, nullable=False)
