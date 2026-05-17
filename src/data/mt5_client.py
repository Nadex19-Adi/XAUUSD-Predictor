import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from src.core.config import settings

class MT5Client:
    def __init__(self):
        self.symbol = settings.SYMBOL
        self.connected = False
        
    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        if not mt5.initialize():
            print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            return False
            
        # Optional: Login if credentials are provided
        if settings.MT5_LOGIN and settings.MT5_PASSWORD and settings.MT5_SERVER:
            authorized = mt5.login(
                settings.MT5_LOGIN, 
                password=settings.MT5_PASSWORD, 
                server=settings.MT5_SERVER
            )
            if not authorized:
                print(f"MT5 failed to connect at account #{settings.MT5_LOGIN}, error code: {mt5.last_error()}")
                return False
                
        self.connected = True
        return True
        
    def get_live_data(self, timeframe=mt5.TIMEFRAME_M5, num_bars=500) -> pd.DataFrame:
        """Fetch latest OHLCV data from MT5."""
        if not self.connected:
            self.connect()
            
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, num_bars)
        if rates is None:
            print(f"Failed to fetch rates, error code: {mt5.last_error()}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # Rename MT5 columns to match yfinance format
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df

    def send_order(self, action: str, volume: float = 0.01):
        """Send a basic market order."""
        if not self.connected:
            return False
            
        order_type = mt5.ORDER_TYPE_BUY if action.lower() == 'buy' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if action.lower() == 'buy' else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI RAG Prediction",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
