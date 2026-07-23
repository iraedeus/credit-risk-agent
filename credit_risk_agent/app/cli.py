import argparse

import pandas as pd

from credit_risk_agent.agent.tools import get_client_financial_metrics, run_model
from credit_risk_agent.config import ARTIFACTS_PATH


def get_client_info(client_id: int) -> None:
    print(run_model(client_id))
    print(get_client_financial_metrics(client_id))


def prompt_agent(prompt: str, verbose: bool = False) -> None:
    pass


def chat_agent(verbose: bool = False) -> None:
    pass


def list_test_clients(limit: int = 10) -> None:
    test_clients = pd.read_csv(ARTIFACTS_PATH / "test_clients.csv")
    ids = test_clients["client_id"].head(limit).astype(str).to_list()
    print(f"Доступные ID клиентов (первые {len(ids)}):")
    print(", ".join(ids))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Command line interface для AI-агента кредитного скоринга и оценки рисков дефолта заёмщиков"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--get-client-info", "-c", type=int, help="Вывести отчёт для конкретного клиента по ID")
    group.add_argument("--prompt", "-p", type=str, help="Задать один вопрос агенту")
    group.add_argument("--chat", action="store_true", help="Начать чат с агентом")
    group.add_argument("--list-clients", action="store_true", help="Вывести ID первых 10 доступных клиентов")

    parser.add_argument("--verbose", "-v", action="store_true", help="Показывать внутренние рассуждения агента")

    args = parser.parse_args()

    if args.verbose and not (args.prompt or args.chat):
        parser.error("Флаг --verbose (-v) доступен только с --prompt или --chat.")

    if args.get_client_info:
        get_client_info(args.get_client_info)
    elif args.prompt:
        prompt_agent(args.prompt, args.verbose)
    elif args.chat:
        chat_agent(args.verbose)
    elif args.list_clients:
        list_test_clients()


if __name__ == "__main__":
    main()
