"""
Unit tests for data schemas validation.
"""

import pytest
from pydantic import ValidationError

from credit_risk_agent.schemas.data_schemas import (
    ClientFinancialMetrics,
    ClientFullInfo,
    ClientPaymentHistory,
    ClientProfile,
)
from credit_risk_agent.schemas.enums import Education, Marriage, Sex


class TestDataSchemas:
    def test_client_financial_metrics_valid(self) -> None:
        metrics = ClientFinancialMetrics(
            client_id=1,
            limit_bal=50000.0,
            avg_bill=12000.5,
            avg_utilization=24.0,
            max_utilization=35.0,
            avg_pay=5000.0,
            repayment_rate=41.6,
            max_delay_status=1,
            delay_months_count=2,
        )
        assert metrics.client_id == 1
        assert metrics.limit_bal == 50000.0
        assert metrics.avg_bill == 12000.5
        assert metrics.delay_months_count == 2

    def test_client_financial_metrics_zero_or_negative_bill(self) -> None:
        """Verify that zero or negative avg_bill does not fail validation."""
        metrics = ClientFinancialMetrics(
            client_id=2,
            limit_bal=100000.0,
            avg_bill=0.0,
            avg_utilization=0.0,
            max_utilization=0.0,
            avg_pay=0.0,
            repayment_rate=0.0,
            max_delay_status=-1,
            delay_months_count=0,
        )
        assert metrics.avg_bill == 0.0

    def test_client_financial_metrics_invalid_delay_months_count(self) -> None:
        """Verify delay_months_count cannot exceed 6 or be negative."""
        with pytest.raises(ValidationError):
            ClientFinancialMetrics(
                client_id=3,
                limit_bal=100000.0,
                avg_bill=1000.0,
                avg_utilization=1.0,
                max_utilization=1.0,
                avg_pay=1000.0,
                repayment_rate=100.0,
                max_delay_status=1,
                delay_months_count=7,
            )

    def test_client_full_info_valid(self) -> None:
        profile = ClientProfile(
            client_id=10,
            limit_bal=50000.0,
            age=30,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.SINGLE,
        )
        history = [
            ClientPaymentHistory(
                client_id=10,
                month=1,
                pay_status=0,
                bill_amt=1000.0,
                pay_amt=1000.0,
            )
        ]
        metrics = ClientFinancialMetrics(
            client_id=10,
            limit_bal=50000.0,
            avg_bill=1000.0,
            avg_utilization=2.0,
            max_utilization=2.0,
            avg_pay=1000.0,
            repayment_rate=100.0,
            max_delay_status=0,
            delay_months_count=0,
        )
        full_info = ClientFullInfo(profile=profile, history=history, metrics=metrics)
        assert full_info.profile.client_id == 10
        assert full_info.metrics.avg_utilization == 2.0

    def test_client_full_info_mismatched_history_client_id(self) -> None:
        profile = ClientProfile(
            client_id=10,
            limit_bal=50000.0,
            age=30,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.SINGLE,
        )
        history = [
            ClientPaymentHistory(
                client_id=99,
                month=1,
                pay_status=0,
                bill_amt=1000.0,
                pay_amt=1000.0,
            )
        ]
        metrics = ClientFinancialMetrics(
            client_id=10,
            limit_bal=50000.0,
            avg_bill=1000.0,
            avg_utilization=2.0,
            max_utilization=2.0,
            avg_pay=1000.0,
            repayment_rate=100.0,
            max_delay_status=0,
            delay_months_count=0,
        )
        with pytest.raises(ValidationError, match="Mismatched history client_id"):
            ClientFullInfo(profile=profile, history=history, metrics=metrics)

    def test_client_full_info_mismatched_metrics_client_id(self) -> None:
        profile = ClientProfile(
            client_id=10,
            limit_bal=50000.0,
            age=30,
            sex=Sex.MALE,
            education=Education.UNIVERSITY,
            marriage=Marriage.SINGLE,
        )
        history = [
            ClientPaymentHistory(
                client_id=10,
                month=1,
                pay_status=0,
                bill_amt=1000.0,
                pay_amt=1000.0,
            )
        ]
        metrics = ClientFinancialMetrics(
            client_id=77,
            limit_bal=50000.0,
            avg_bill=1000.0,
            avg_utilization=2.0,
            max_utilization=2.0,
            avg_pay=1000.0,
            repayment_rate=100.0,
            max_delay_status=0,
            delay_months_count=0,
        )
        with pytest.raises(ValidationError, match="Mismatched metrics client_id"):
            ClientFullInfo(profile=profile, history=history, metrics=metrics)
