from datetime import datetime
import yfinance as yf
import pytz


def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """사용자가 요청한 지역(타임존)의 현재 날짜와 시간을 반환합니다.

    Args:
        timezone: 정보를 가져올 지역 타임존명 (예: 'Asia/Seoul', 'America/New_York', 'Europe/London', 'UTC' 등)
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    now_timezone = f"{now} ({timezone})"
    print(now_timezone)
    return now_timezone


def get_yf_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    print(info)
    return info


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "사용자가 원하는 국가의 현재 시간을 알려줍니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "현재 날짜와 시간을 반환할 타임존을 입력하세요. (예시: Asia/Seoul, America/New_York)",
                    },
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_yf_stock_info",
            "description": "해당 종목의 Yahoo Finance 정보를 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Yahoo Finance 정보를 반환할 종목의 티커를 입력하세요. (예: AAPL)",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]

if __name__ == "__main__":
    get_current_time("America/New_York")
    info = get_yf_stock_info("AAPL")
