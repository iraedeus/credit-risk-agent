"""
SQLAlchemy ORM models for the Data Service database tables.
"""

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """
    Base class for declarative SQLAlchemy ORM models.
    """

    pass


class ClientDB(Base):
    """
    Database entity representing client demographic profile and credit limit.

    Attributes
    ----------
    client_id : int
        Unique identifier for the client (primary key).
    limit_bal : float
        Credit limit balance allocated to the client.
    sex : int
        Gender indicator (1: male, 2: female).
    education : int
        Education level classification code.
    marriage : int
        Marital status classification code.
    age : int
        Age of the client in years.
    history : list of PaymentHistoryDB
        List of related payment history records for the client.
    """

    __tablename__ = "clients"

    client_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    limit_bal: Mapped[float] = mapped_column(Float, nullable=False)
    sex: Mapped[int] = mapped_column(Integer, nullable=False)
    education: Mapped[int] = mapped_column(Integer, nullable=False)
    marriage: Mapped[int] = mapped_column(Integer, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)

    history: Mapped[list["PaymentHistoryDB"]] = relationship(
        "PaymentHistoryDB", back_populates="client", cascade="all, delete-orphan"
    )


class PaymentHistoryDB(Base):
    """
    Database entity representing a single month's payment statement record.

    Attributes
    ----------
    client_id : int
        Foreign key referencing `clients.client_id` (composite primary key).
    month : int
        Month index within the historical period (1 to 6) (composite primary key).
    pay_status : int
        Repayment delay status (-1: pay duly, 1..8: payment delay in months).
    bill_amt : float
        Bill statement amount for the given month.
    pay_amt : float
        Amount paid in the given month.
    client : ClientDB
        Parent client entity associated with this payment history entry.
    """

    __tablename__ = "payment_history"

    client_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clients.client_id", ondelete="CASCADE"), primary_key=True
    )
    month: Mapped[int] = mapped_column(Integer, primary_key=True)
    pay_status: Mapped[int] = mapped_column(Integer, nullable=False)
    bill_amt: Mapped[float] = mapped_column(Float, nullable=False)
    pay_amt: Mapped[float] = mapped_column(Float, nullable=False)

    client: Mapped[ClientDB] = relationship(ClientDB, back_populates="history")
