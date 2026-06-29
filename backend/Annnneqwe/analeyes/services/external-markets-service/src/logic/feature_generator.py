from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
try:
    import pandas_ta as ta
except Exception:
    try:
        import pandas_ta_classic as ta
    except Exception:
        ta = None
try:
    import talib
except Exception:
    talib = None

@dataclass(slots=True)
class FeatureGeneratorConfig:
    breakout_lookback: int = 20
    volume_window: int = 20
    ema_fast_period: int = 8
    ema_slow_period: int = 21

class FeatureGenerator:

    def __init__(self, config: dict[str, Any] | None=None) -> None:
        cfg = config or {}
        fast_period = max(1, int(cfg.get('ema_fast_period', 8)))
        slow_period = max(2, int(cfg.get('ema_slow_period', 21)))
        if slow_period <= fast_period:
            slow_period = fast_period + 1
        self.config = FeatureGeneratorConfig(breakout_lookback=int(cfg.get('breakout_lookback', 20)), volume_window=int(cfg.get('volume_window', 20)), ema_fast_period=fast_period, ema_slow_period=slow_period)

    def compute_indicators(self, frame: pd.DataFrame, asset_class: str='stock', benchmark_close: pd.Series | None=None, dxy_close: pd.Series | None=None) -> pd.DataFrame:
        df = frame.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f'Missing input column: {col}')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        fast_period = self.config.ema_fast_period
        slow_period = self.config.ema_slow_period
        df['ema_fast'] = self._ema(df['close'], fast_period)
        df['ema_slow'] = self._ema(df['close'], slow_period)
        df['ema_8'] = df['ema_fast'] if fast_period == 8 else self._ema(df['close'], 8)
        df['ema_21'] = df['ema_slow'] if slow_period == 21 else self._ema(df['close'], 21)
        df['ema_50'] = self._ema(df['close'], 50)
        df['ema_200'] = self._ema(df['close'], 200)
        macd_line, macd_signal, macd_hist = self._macd(df['close'])
        df['macd_line'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['adx'] = self._adx(df['high'], df['low'], df['close'], 14)
        df['supertrend'] = self._supertrend(df['high'], df['low'], df['close'], 10, 3.0)
        df['rsi'] = self._rsi(df['close'], 14)
        stoch_k, stoch_d = self._stoch(df['high'], df['low'], df['close'], 14, 3, 3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        bb_upper, bb_middle, bb_lower = self._bollinger(df['close'], 20, 2.0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
        df['atr'] = self._atr(df['high'], df['low'], df['close'], 14)
        df['atr_pct'] = df['atr'] / df['close'].replace(0, np.nan)
        df['vwap'] = self._vwap(frame=df, asset_class=asset_class)
        df['vol_rel'] = self._volume_relative_by_slot(frame=df, asset_class=asset_class)
        df['obv'] = self._obv(df['close'], df['volume'])
        lookback = self.config.breakout_lookback
        df['support_nearest'] = df['low'].rolling(lookback).min().shift(1)
        df['resistance_nearest'] = df['high'].rolling(lookback).max().shift(1)
        df['price_pct'] = df['close'].pct_change() * 100.0
        if benchmark_close is not None:
            df['sp500_correlation'] = self._rolling_corr(df['close'], benchmark_close, 20)
        else:
            df['sp500_correlation'] = np.nan
        if dxy_close is not None:
            df['dxy_correlation'] = self._rolling_corr(df['close'], dxy_close, 20)
        else:
            df['dxy_correlation'] = np.nan
        df['ema_fast'] = self._enforce_warmup(df['ema_fast'], fast_period)
        df['ema_slow'] = self._enforce_warmup(df['ema_slow'], slow_period)
        df['ema_8'] = self._enforce_warmup(df['ema_8'], 8)
        df['ema_21'] = self._enforce_warmup(df['ema_21'], 21)
        df['ema_50'] = self._enforce_warmup(df['ema_50'], 50)
        df['ema_200'] = self._enforce_warmup(df['ema_200'], 200)
        df['macd_line'] = self._enforce_warmup(df['macd_line'], 26)
        df['macd_signal'] = self._enforce_warmup(df['macd_signal'], 34)
        df['macd_hist'] = self._enforce_warmup(df['macd_hist'], 34)
        df['rsi'] = self._enforce_warmup(df['rsi'], 14)
        df['adx'] = self._enforce_warmup(df['adx'], 27)
        df['supertrend'] = self._enforce_warmup(df['supertrend'], 10)
        df['stoch_k'] = self._enforce_warmup(df['stoch_k'], 16)
        df['stoch_d'] = self._enforce_warmup(df['stoch_d'], 18)
        df['bb_upper'] = self._enforce_warmup(df['bb_upper'], 20)
        df['bb_middle'] = self._enforce_warmup(df['bb_middle'], 20)
        df['bb_lower'] = self._enforce_warmup(df['bb_lower'], 20)
        df['bb_width'] = self._enforce_warmup(df['bb_width'], 20)
        df['atr'] = self._enforce_warmup(df['atr'], 14)
        df['atr_pct'] = self._enforce_warmup(df['atr_pct'], 14)
        df['sp500_correlation'] = self._enforce_warmup(df['sp500_correlation'], 20)
        df['dxy_correlation'] = self._enforce_warmup(df['dxy_correlation'], 20)
        df[f'ema_{fast_period}'] = df['ema_fast']
        df[f'ema_{slow_period}'] = df['ema_slow']
        return df

    def build_feature_payload(self, indicator_df: pd.DataFrame, asset_class: str, bid: float | None, ask: float | None) -> dict[str, Any]:
        last = indicator_df.iloc[-1]
        market_state = self._market_state(last)
        payload: dict[str, Any] = {'current_price': self._safe(last.get('close')), 'price_pct': self._safe(last.get('price_pct')), 'market_state': market_state, 'rsi': self._safe(last.get('rsi')), 'macd': self._safe(last.get('macd_line')), 'macd_signal': self._safe(last.get('macd_signal')), 'macd_hist': self._safe(last.get('macd_hist')), 'ema_short': self._safe(last.get('ema_fast', last.get('ema_8'))), 'ema_long': self._safe(last.get('ema_slow', last.get('ema_21'))), 'ema_50': self._safe(last.get('ema_50')), 'ema_200': self._safe(last.get('ema_200')), 'adx': self._safe(last.get('adx')), 'supertrend': self._safe(last.get('supertrend')), 'stoch_k': self._safe(last.get('stoch_k')), 'stoch_d': self._safe(last.get('stoch_d')), 'bb_upper': self._safe(last.get('bb_upper')), 'bb_middle': self._safe(last.get('bb_middle')), 'bb_lower': self._safe(last.get('bb_lower')), 'bb_width': self._safe(last.get('bb_width')), 'atr': self._safe(last.get('atr')), 'atr_pct': self._safe(last.get('atr_pct')), 'vwap': self._safe(last.get('vwap')), 'vol_rel': self._safe(last.get('vol_rel')), 'obv': self._safe(last.get('obv')), 'support_nearest': self._safe(last.get('support_nearest')), 'resistance_nearest': self._safe(last.get('resistance_nearest')), 'sp500_correlation': self._safe(last.get('sp500_correlation')), 'dxy_correlation': self._safe(last.get('dxy_correlation')), 'bid_ask_spread_pips': None}
        if asset_class == 'forex' and bid is not None and (ask is not None):
            pip_multiplier = 100 if 'JPY' in str(indicator_df.attrs.get('symbol', '')) else 10000
            payload['bid_ask_spread_pips'] = round(abs(ask - bid) * pip_multiplier, 4)
        weekday = indicator_df.index[-1].weekday()
        dow_keys = ['dow_monday', 'dow_tuesday', 'dow_wednesday', 'dow_thursday', 'dow_friday', 'dow_saturday', 'dow_sunday']
        for idx, key in enumerate(dow_keys):
            payload[key] = 1.0 if idx == weekday else 0.0
        return payload

    @staticmethod
    def build_trend_context(indicator_df: pd.DataFrame) -> dict[str, Any]:
        if indicator_df.empty:
            return {'direction': 'NEUTRAL', 'strength': 0.0}
        last = indicator_df.iloc[-1]
        ema_fast = last.get('ema_fast', last.get('ema_8'))
        ema_slow = last.get('ema_slow', last.get('ema_21'))
        close = last.get('close')
        adx = last.get('adx')
        if pd.isna([ema_fast, ema_slow]).any():
            return {'direction': 'NEUTRAL', 'strength': 0.0}
        direction = 'NEUTRAL'
        if float(ema_fast) > float(ema_slow):
            direction = 'LONG'
        elif float(ema_fast) < float(ema_slow):
            direction = 'SHORT'
        if direction == 'NEUTRAL' or pd.isna(close) or float(close) == 0.0:
            return {'direction': direction, 'strength': 0.0}
        base_strength = min(abs(float(ema_fast) - float(ema_slow)) / max(abs(float(close)), 1e-09) * 25.0, 1.0)
        adx_weight = 0.5
        if pd.notna(adx):
            adx_weight = min(max(float(adx), 0.0) / 50.0, 1.0)
        return {'direction': direction, 'strength': round(min(base_strength * adx_weight, 1.0), 4)}

    @staticmethod
    def build_historical_snapshots(indicator_df: pd.DataFrame, count: int=2) -> list[dict[str, Any]]:
        if indicator_df.empty or count <= 0:
            return []
        history = indicator_df.tail(count)
        snapshots: list[dict[str, Any]] = []
        for ts, row in history.iloc[::-1].iterrows():
            snapshots.append({'timestamp': ts.isoformat().replace('+00:00', 'Z'), 'close': FeatureGenerator._safe(row.get('close')), 'volume': FeatureGenerator._safe(row.get('volume')), 'rsi': FeatureGenerator._safe(row.get('rsi')), 'vol_rel': FeatureGenerator._safe(row.get('vol_rel')), 'macd_hist': FeatureGenerator._safe(row.get('macd_hist'))})
        return snapshots

    @staticmethod
    def _market_state(last: pd.Series) -> str:
        ema_short = last.get('ema_fast', last.get('ema_8'))
        ema_long = last.get('ema_slow', last.get('ema_21'))
        adx = last.get('adx')
        if pd.notna(ema_short) and pd.notna(ema_long) and pd.notna(adx) and (adx >= 25):
            if ema_short > ema_long:
                return 'TRENDING_UP'
            if ema_short < ema_long:
                return 'TRENDING_DOWN'
        return 'RANGING'

    @staticmethod
    def _safe(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (float, np.floating, int, np.integer)):
            if np.isnan(value):
                return None
            return float(value)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(numeric) else float(numeric)

    @staticmethod
    def _rolling_corr(base: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
        aligned = pd.concat([base, benchmark], axis=1, join='inner').dropna()
        if aligned.empty:
            return pd.Series(index=base.index, dtype='float64')
        corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
        return corr.reindex(base.index)

    @staticmethod
    def _enforce_warmup(series: pd.Series, min_periods: int) -> pd.Series:
        if min_periods <= 1 or series.empty:
            return series
        out = series.copy()
        cutoff = min(len(out), min_periods - 1)
        out.iloc[:cutoff] = np.nan
        return out

    @staticmethod
    def _ensure_utc_index(index: pd.Index) -> pd.DatetimeIndex:
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError('FeatureGenerator expects DatetimeIndex in OHLCV frame')
        if index.tz is None:
            return index.tz_localize('UTC')
        return index.tz_convert('UTC')

    @staticmethod
    def _slot_labels(index_utc: pd.DatetimeIndex, asset_class: str) -> pd.Series:
        if asset_class in {'stock', 'index'}:
            localized = index_utc.tz_convert('America/New_York')
        else:
            localized = index_utc
        return pd.Series(localized.strftime('%H:%M'), index=index_utc)

    @staticmethod
    def _session_labels(index_utc: pd.DatetimeIndex, asset_class: str) -> pd.Series:
        if asset_class in {'stock', 'index'}:
            localized = index_utc.tz_convert('America/New_York')
            return pd.Series(localized.date, index=index_utc, dtype='object')
        if asset_class == 'forex':
            shifted = index_utc - pd.Timedelta(hours=22)
            return pd.Series(shifted.date, index=index_utc, dtype='object')
        return pd.Series(index_utc.date, index=index_utc, dtype='object')

    def _volume_relative_by_slot(self, frame: pd.DataFrame, asset_class: str) -> pd.Series:
        window = max(int(self.config.volume_window), 1)
        index_utc = self._ensure_utc_index(frame.index)
        slot_labels = self._slot_labels(index_utc=index_utc, asset_class=asset_class)
        volume = frame['volume'].astype('float64')
        slot_baseline = volume.groupby(slot_labels, sort=False).transform(lambda group: group.shift(1).rolling(window=window, min_periods=window).mean())
        rolling_baseline = volume.shift(1).rolling(window=window, min_periods=window).mean()
        baseline = slot_baseline.where(slot_baseline.notna(), rolling_baseline)
        return volume / baseline.replace(0, np.nan)

    def _vwap(self, frame: pd.DataFrame, asset_class: str) -> pd.Series:
        index_utc = self._ensure_utc_index(frame.index)
        session_labels = self._session_labels(index_utc=index_utc, asset_class=asset_class)
        typical = (frame['high'] + frame['low'] + frame['close']) / 3.0
        volume = frame['volume'].astype('float64')
        cumulative_value = (typical * volume).groupby(session_labels, sort=False).cumsum()
        cumulative_volume = volume.groupby(session_labels, sort=False).cumsum().replace(0, np.nan)
        return cumulative_value / cumulative_volume

    @staticmethod
    def _ema(series: pd.Series, length: int) -> pd.Series:
        if ta is not None:
            result = ta.ema(series, length=length)
            if result is not None:
                return result
        if talib is not None:
            values = talib.EMA(series.to_numpy(dtype='float64'), timeperiod=length)
            return pd.Series(values, index=series.index)
        return series.ewm(span=length, adjust=False, min_periods=length).mean()

    @staticmethod
    def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        if ta is not None:
            macd = ta.macd(series, fast=12, slow=26, signal=9)
            if macd is not None and (not macd.empty):
                line = macd.iloc[:, 0]
                hist = macd.iloc[:, 1]
                signal = macd.iloc[:, 2]
                return (line, signal, hist)
        if talib is not None:
            line, signal, hist = talib.MACD(series.to_numpy(dtype='float64'), fastperiod=12, slowperiod=26, signalperiod=9)
            return (pd.Series(line, index=series.index), pd.Series(signal, index=series.index), pd.Series(hist, index=series.index))
        ema_fast = series.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow = series.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_hist = macd_line - macd_signal
        return (macd_line, macd_signal, macd_hist)

    @staticmethod
    def _rsi(series: pd.Series, length: int) -> pd.Series:
        if ta is not None:
            result = ta.rsi(series, length=length)
            if result is not None:
                return result
        if talib is not None:
            values = talib.RSI(series.to_numpy(dtype='float64'), timeperiod=length)
            return pd.Series(values, index=series.index)
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=length, min_periods=length).mean()
        avg_loss = loss.rolling(window=length, min_periods=length).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
        rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
        return rsi

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        if ta is not None:
            result = ta.adx(high, low, close, length=length)
            if result is not None and (not result.empty):
                return result.iloc[:, 0]
        if talib is not None:
            values = talib.ADX(high.to_numpy(dtype='float64'), low.to_numpy(dtype='float64'), close.to_numpy(dtype='float64'), timeperiod=length)
            return pd.Series(values, index=close.index)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr_components = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1)
        tr = tr_components.max(axis=1)
        atr = tr.rolling(length).mean()
        plus_di = 100 * (plus_dm.rolling(length).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(length).mean() / atr.replace(0, np.nan))
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        return dx.rolling(length).mean()

    @staticmethod
    def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, length: int, multiplier: float) -> pd.Series:
        if ta is not None:
            result = ta.supertrend(high, low, close, length=length, multiplier=multiplier)
            if result is not None and (not result.empty):
                return result.iloc[:, 0]
        hl2 = (high + low) / 2.0
        atr = FeatureGenerator._atr(high, low, close, length)
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        trend = pd.Series(np.nan, index=close.index, dtype='float64')
        first_valid = atr.first_valid_index()
        if first_valid is None:
            return trend
        start = close.index.get_loc(first_valid)
        trend.iloc[start] = close.iloc[start]
        for i in range(start + 1, len(close)):
            prev = trend.iloc[i - 1]
            if pd.isna(prev):
                prev = close.iloc[i - 1]
            if pd.isna(upper.iloc[i - 1]) or pd.isna(lower.iloc[i - 1]):
                continue
            if close.iloc[i] > upper.iloc[i - 1]:
                trend.iloc[i] = lower.iloc[i]
            elif close.iloc[i] < lower.iloc[i - 1]:
                trend.iloc[i] = upper.iloc[i]
            else:
                trend.iloc[i] = prev
        return trend

    @staticmethod
    def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int, smooth_k: int) -> tuple[pd.Series, pd.Series]:
        if ta is not None:
            result = ta.stoch(high, low, close, k=k, d=d, smooth_k=smooth_k)
            if result is not None and (not result.empty):
                return (result.iloc[:, 0], result.iloc[:, 1])
        if talib is not None:
            k_values, d_values = talib.STOCH(high.to_numpy(dtype='float64'), low.to_numpy(dtype='float64'), close.to_numpy(dtype='float64'), fastk_period=k, slowk_period=smooth_k, slowk_matype=0, slowd_period=d, slowd_matype=0)
            return (pd.Series(k_values, index=close.index), pd.Series(d_values, index=close.index))
        ll = low.rolling(k).min()
        hh = high.rolling(k).max()
        k_line = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
        k_smooth = k_line.rolling(smooth_k).mean()
        d_line = k_smooth.rolling(d).mean()
        return (k_smooth, d_line)

    @staticmethod
    def _bollinger(close: pd.Series, length: int, std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
        if ta is not None:
            result = ta.bbands(close, length=length, std=std)
            if result is not None and (not result.empty):
                upper_col = next((col for col in result.columns if str(col).startswith('BBU')), None)
                middle_col = next((col for col in result.columns if str(col).startswith('BBM')), None)
                lower_col = next((col for col in result.columns if str(col).startswith('BBL')), None)
                if upper_col and middle_col and lower_col:
                    return (result[upper_col], result[middle_col], result[lower_col])
                return (result.iloc[:, 2], result.iloc[:, 1], result.iloc[:, 0])
        if talib is not None:
            upper, middle, lower = talib.BBANDS(close.to_numpy(dtype='float64'), timeperiod=length, nbdevup=std, nbdevdn=std, matype=0)
            return (pd.Series(upper, index=close.index), pd.Series(middle, index=close.index), pd.Series(lower, index=close.index))
        middle = close.rolling(length).mean()
        stdev = close.rolling(length).std(ddof=0)
        upper = middle + stdev * std
        lower = middle - stdev * std
        return (upper, middle, lower)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        if ta is not None:
            result = ta.atr(high, low, close, length=length)
            if result is not None:
                return result
        if talib is not None:
            values = talib.ATR(high.to_numpy(dtype='float64'), low.to_numpy(dtype='float64'), close.to_numpy(dtype='float64'), timeperiod=length)
            return pd.Series(values, index=close.index)
        tr_components = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1)
        tr = tr_components.max(axis=1)
        return tr.rolling(length).mean()

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        if ta is not None:
            result = ta.obv(close, volume)
            if result is not None:
                return result
        if talib is not None:
            values = talib.OBV(close.to_numpy(dtype='float64'), volume.to_numpy(dtype='float64'))
            return pd.Series(values, index=close.index)
        direction = np.sign(close.diff().fillna(0.0))
        return (direction * volume).cumsum()
