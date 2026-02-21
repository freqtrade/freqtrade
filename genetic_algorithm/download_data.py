#!/usr/bin/env python3
"""
Script to download real market data from exchanges for backtesting.

This script uses FreqTrade's data download functionality to fetch
real OHLCV data from exchanges for use in strategy backtesting.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(
    pairs: list = None,
    timeframes: list = None,
    exchange: str = "binance",
    days: int = 90,
    datadir: Path = None
):
    """
    Download real market data from exchange.
    
    Args:
        pairs: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        timeframes: List of timeframes (e.g., ['5m', '1h'])
        exchange: Exchange name (default: binance)
        days: Number of days of historical data to download
        datadir: Directory to save data (default: user_data/data/<exchange>)
    """
    from freqtrade.configuration import Configuration
    from freqtrade.data.history import download_data_main
    from freqtrade.resolvers import ExchangeResolver
    
    # Default values
    if pairs is None:
        pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    
    if timeframes is None:
        timeframes = ['5m', '15m', '1h']
    
    if datadir is None:
        datadir = Path(__file__).parent.parent / "user_data" / "data" / exchange
    
    datadir.mkdir(parents=True, exist_ok=True)
    
    # Calculate timerange
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    
    logger.info(f"Downloading data from {exchange}")
    logger.info(f"Pairs: {pairs}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Timerange: {timerange}")
    logger.info(f"Data directory: {datadir}")
    
    # Create configuration for data download
    config = {
        "exchange": {
            "name": exchange,
            "key": "",
            "secret": "",
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": pairs,
        },
        "datadir": datadir,
        "user_data_dir": Path(__file__).parent.parent / "user_data",
        "timeframes": timeframes,
        "timerange": timerange,
        "pairs": pairs,
        "trading_mode": "spot",
        "margin_mode": "",
        "download_trades": False,  # Only download candle data
        "days": days,
        "erase": False,  # Don't erase existing data
        "dry_run": False,  # Not dry run for data download
        "stake_currency": "USDT",
        "runmode": "other",  # Required by FreqTrade
    }
    
    try:
        # Download data using FreqTrade's built-in downloader
        logger.info("Starting data download...")
        
        # Note: We'll use a simpler approach with the exchange directly
        from freqtrade.exchange import Exchange
        from freqtrade.configuration import TimeRange as FTTimeRange
        
        exchange_obj = ExchangeResolver.load_exchange(config)
        
        # Convert the calculated timerange string to a TimeRange object so the
        # download covers the full requested history rather than only the default
        # number of candles the exchange returns without an explicit range.
        ft_timerange = FTTimeRange.parse_timerange(timerange)
        
        for timeframe in timeframes:
            logger.info(f"Downloading {timeframe} data...")
            for pair in pairs:
                try:
                    logger.info(f"  Downloading {pair}...")
                    
                    # Download the data
                    from freqtrade.data.history import load_pair_history, refresh_backtest_ohlcv_data
                    
                    refresh_backtest_ohlcv_data(
                        exchange=exchange_obj,
                        pairs=[pair],
                        timeframes=[timeframe],
                        datadir=datadir,
                        timerange=ft_timerange,
                        erase=False
                    )
                    
                    logger.info(f"  ✓ Downloaded {pair} {timeframe}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to download {pair} {timeframe}: {e}")
        
        logger.info("Data download completed!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=" * 80)
    print("FreqTrade GA - Market Data Downloader")
    print("=" * 80)
    print()
    
    # Default configuration - can be modified based on needs
    pairs = [
        'BTC/USDT',
        'ETH/USDT',
        'BNB/USDT',
        'SOL/USDT',
        'ADA/USDT'
    ]
    
    timeframes = ['5m', '15m', '1h']
    exchange = 'binance'
    days = 90  # 3 months of data
    
    print(f"Configuration:")
    print(f"  Exchange: {exchange}")
    print(f"  Pairs: {', '.join(pairs)}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Days of history: {days}")
    print()
    
    success = download_data(
        pairs=pairs,
        timeframes=timeframes,
        exchange=exchange,
        days=days
    )
    
    if success:
        print()
        print("=" * 80)
        print("✓ Data download completed successfully!")
        print("=" * 80)
        print()
        print("The downloaded data can now be used for backtesting strategies.")
        print()
        return 0
    else:
        print()
        print("=" * 80)
        print("✗ Data download failed!")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
