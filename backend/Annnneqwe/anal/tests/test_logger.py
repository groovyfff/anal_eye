import json
from src.utils.logger import get_logger, setup_logging

def test_setup_logging_respects_configured_format(capsys) -> None:
    setup_logging({'level': 'INFO', 'service_name': 'svc', 'format': '%(levelname)s|%(message)s|%(service_name)s|%(symbol)s'})
    logger = get_logger(__name__)
    logger.info('hello', extra={'symbol': 'AAPL'})
    output = capsys.readouterr().out.strip()
    assert output == 'INFO|hello|svc|AAPL'

def test_setup_logging_injects_required_context_fields(capsys) -> None:
    setup_logging({'level': 'INFO', 'service_name': 'svc', 'format': '%(levelname)s|%(message)s'})
    logger = get_logger(__name__)
    logger.info('hello', extra={'symbol': 'AAPL'})
    output = capsys.readouterr().out.strip()
    assert output == '[svc][AAPL] INFO|hello'

def test_setup_logging_json_mode(capsys) -> None:
    setup_logging({'level': 'INFO', 'service_name': 'svc', 'json': True})
    logger = get_logger(__name__)
    logger.info('hello', extra={'symbol': 'AAPL'})
    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload['service_name'] == 'svc'
    assert payload['symbol'] == 'AAPL'
    assert payload['message'] == 'hello'
    assert payload['level'] == 'INFO'
