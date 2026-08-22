"""
Command Line Interface (CLI) application for interacting with Credit Risk Agent.
"""

import argparse
import os

from dotenv import load_dotenv
from gigachat import GigaChat

from credit_risk_agent.agent.agent import CreditRiskAgent
from credit_risk_agent.agent.tools import get_client_financial_metrics, run_model
from credit_risk_agent.config import GIGACHAT_MODEL
from credit_risk_agent.services.data_service.client import get_data_service_client


def get_client_info(client_id: int) -> None:
    """
    Print model prediction and financial metrics report for a specified client ID.

    Parameters
    ----------
    client_id : int
        The unique identifier of the client.
    """
    print(run_model(client_id))
    print(get_client_financial_metrics(client_id))


def prompt_agent(prompt: str, verbose: bool = False) -> None:
    """
    Send a single prompt query to the CreditRiskAgent and print the response.

    Parameters
    ----------
    prompt : str
        The query string for the agent.
    verbose : bool, default=False
        Whether to print intermediate ReAct thoughts and tool observations.
    """
    load_dotenv()
    credentials = os.getenv("GIGACHAT_CREDENTIALS")

    if not credentials or credentials == "your_gigachat_authorization_data":
        print("Добавьте пожалуйста ваш API-ключ для GigaChat в файл .env")
        return

    with GigaChat(credentials=credentials, model=GIGACHAT_MODEL, verify_ssl_certs=False) as client:
        agent = CreditRiskAgent(client)
        response = agent.run(user_prompt=prompt, verbose=verbose)
        print("\n" + response)


def chat_agent(verbose: bool = False) -> None:
    """
    Start an interactive multi-turn REPL chat session with CreditRiskAgent.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print intermediate thoughts and tool executions.
    """
    load_dotenv()
    credentials = os.getenv("GIGACHAT_CREDENTIALS")

    if not credentials or credentials == "your_gigachat_authorization_data":
        print("Добавьте пожалуйста ваш API-ключ для GigaChat в файл .env")
        return

    with GigaChat(credentials=credentials, model=GIGACHAT_MODEL, verify_ssl_certs=False) as client:
        agent = CreditRiskAgent(client, max_iterations=25)
        while True:
            user_input = input("Вы > ")
            if user_input == "clear":
                agent.clear_history()
                print("История очищена.")
                continue

            if user_input == "exit":
                break

            response = agent.run(user_input, verbose=verbose)
            print(f"Agent > {response}")


def list_test_clients(limit: int = 10) -> None:
    """
    Display a list of available test client IDs from dataset artifacts.

    Parameters
    ----------
    limit : int, default=10
        Maximum number of client IDs to display.
    """
    client_ids = get_data_service_client().get_clients(offset=0, limit=limit)
    print(f"Доступные ID клиентов (первые {len(client_ids)}):")
    print(", ".join(map(str, client_ids)))


def main() -> None:
    """
    Parse command-line arguments and execute chosen CLI mode.
    """
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
